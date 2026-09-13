package me.rerere.rikkahub.data.ai.providers

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.channels.onFailure
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonArrayBuilder
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonObjectBuilder
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlinx.serialization.json.putJsonObject
import me.rerere.ai.core.InputSchema
import me.rerere.ai.core.MessageRole
import me.rerere.ai.core.ReasoningLevel
import me.rerere.ai.core.TokenUsage
import me.rerere.ai.core.Tool
import me.rerere.ai.provider.Model
import me.rerere.ai.provider.Modality
import me.rerere.ai.provider.ModelAbility
import me.rerere.ai.provider.Provider
import me.rerere.ai.provider.ProviderSetting
import me.rerere.ai.provider.TextGenerationParams
import me.rerere.ai.provider.TextGenerationResult
import me.rerere.ai.provider.stream.DecodeResult
import me.rerere.ai.provider.stream.SseEvent
import me.rerere.ai.provider.stream.StreamChunkDecoder
import me.rerere.ai.ui.StreamChunk
import me.rerere.ai.ui.UIMessage
import me.rerere.ai.ui.UIMessagePart
import me.rerere.ai.util.HttpException
import me.rerere.ai.util.KeyRoulette
import me.rerere.ai.util.encodeBase64
import me.rerere.ai.util.mergeCustomBody
import me.rerere.ai.util.parseErrorDetail
import me.rerere.ai.util.stringSafe
import me.rerere.ai.util.toHeaders
import me.rerere.common.http.await
import me.rerere.common.http.jsonArrayOrNull
import me.rerere.common.http.jsonObjectOrNull
import me.rerere.common.http.jsonPrimitiveOrNull
import okhttp3.MediaType
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okhttp3.sse.EventSource
import okhttp3.sse.EventSourceListener
import okhttp3.sse.EventSources
import java.util.Base64
import java.util.UUID
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

private const val TAG = "ZhipuNativeProvider"

private val json = Json {
    ignoreUnknownKeys = true
    explicitNulls = false
}

private val JSON_MEDIA_TYPE: MediaType = "application/json".toMediaType()

private const val DEFAULT_CHAT_PATH = "/chat/completions"

/** 智谱原生 JWT 有效期, 官方建议不超过 3 小时 */
private const val JWT_EXPIRE_DURATION_MS = 60 * 60 * 1000L

private const val HMAC_SHA256 = "HmacSHA256"

/**
 * 智谱 (BigModel / GLM) **原生协议** ChatProvider 实现。
 *
 * 与项目内置的 OpenAI 兼容适配器完全独立, 不复用其请求构造与流式解码逻辑:
 * - 请求体带 `request_id` / `thinking` 等智谱 v4 paas 专有字段;
 * - `temperature` / `top_p` 按智谱取值范围做钳制, 避免直接 400;
 * - 鉴权支持两种原生方式: 直接 `Bearer {API-KEY}`, 或用 `{id}.{secret}` 现场签发 HS256 JWT;
 * - 流式按智谱 SSE 协议解析, 以 `data: [DONE]` 结束。
 */
class ZhipuNativeProvider(
    private val client: OkHttpClient,
    context: Context? = null,
) : Provider<ProviderSetting.Zhipu> {

    private val keyRoulette: KeyRoulette =
        if (context != null) KeyRoulette.lru(context) else KeyRoulette.default()

    // ---------------------------------------------------------------------
    // 模型列表
    // ---------------------------------------------------------------------

    override suspend fun listModels(providerSetting: ProviderSetting.Zhipu): List<Model> =
        withContext(Dispatchers.IO) {
            val remote = runCatching { fetchRemoteModels(providerSetting) }
                .onFailure { Log.w(TAG, "listModels failed, fallback to built-in list", it) }
                .getOrDefault(emptyList())
            remote.ifEmpty { FALLBACK_MODELS.map { Model(modelId = it, displayName = it) } }
        }

    private suspend fun fetchRemoteModels(providerSetting: ProviderSetting.Zhipu): List<Model> {
        val request = Request.Builder()
            .url("${providerSetting.baseUrl.trimEnd('/')}/models")
            .addHeader("Authorization", "Bearer ${authToken(providerSetting)}")
            .get()
            .build()

        val response = client.newCall(request).await()
        val bodyStr = response.body.string()
        if (!response.isSuccessful) {
            throw httpError(response.code, bodyStr, "Failed to get Zhipu models")
        }

        val payload = json.parseToJsonElement(bodyStr).jsonObjectOrNull ?: return emptyList()
        val array = payload["data"]?.jsonArrayOrNull
            ?: payload["models"]?.jsonArrayOrNull
            ?: payload["result"]?.jsonArrayOrNull
            ?: return emptyList()

        return array.mapNotNull { element ->
            val obj = element.jsonObjectOrNull ?: return@mapNotNull null
            val id = obj["id"]?.jsonPrimitiveOrNull?.contentOrNull
                ?: obj["model"]?.jsonPrimitiveOrNull?.contentOrNull
                ?: return@mapNotNull null
            Model(modelId = id, displayName = id)
        }
    }

    // ---------------------------------------------------------------------
    // 非流式生成
    // ---------------------------------------------------------------------

    override suspend fun generateText(
        providerSetting: ProviderSetting.Zhipu,
        messages: List<UIMessage>,
        params: TextGenerationParams,
    ): TextGenerationResult = withContext(Dispatchers.IO) {
        val requestBody = buildRequestBody(providerSetting, messages, params, stream = false)

        val request = Request.Builder()
            .url(endpoint(providerSetting))
            .headers(params.customHeaders.toHeaders())
            .post(requestBody.toString().toRequestBody(JSON_MEDIA_TYPE))
            .addHeader("Authorization", "Bearer ${authToken(providerSetting)}")
            .addHeader("Content-Type", "application/json")
            .build()

        val response = client.newCall(request).await()
        val bodyStr = response.body.string()
        if (!response.isSuccessful) {
            throw httpError(response.code, bodyStr, "Zhipu generateText failed")
        }

        val payload = runCatching { json.parseToJsonElement(bodyStr).jsonObject }
            .getOrElse { throw HttpException("Invalid Zhipu response: $bodyStr") }

        // 智谱在 HTTP 200 里也可能回传业务错误
        if (isErrorPayload(payload)) throw parseNativeError(payload)

        val choice = payload["choices"]?.jsonArrayOrNull?.firstOrNull()?.jsonObjectOrNull
            ?: throw HttpException("Zhipu response has no choices: $bodyStr")
        val messageJson = choice["message"]?.jsonObjectOrNull
            ?: throw HttpException("Zhipu response has no message: $bodyStr")

        TextGenerationResult(
            id = payload["id"]?.jsonPrimitiveOrNull?.contentOrNull
                ?: payload["request_id"]?.jsonPrimitiveOrNull?.contentOrNull.orEmpty(),
            model = payload["model"]?.jsonPrimitiveOrNull?.contentOrNull ?: params.model.modelId,
            message = parseNativeMessage(messageJson),
            finishReason = choice["finish_reason"]?.jsonPrimitiveOrNull?.contentOrNull,
            usage = parseUsage(payload["usage"]?.jsonObjectOrNull),
        )
    }

    // ---------------------------------------------------------------------
    // 流式生成 (智谱原生 SSE)
    // ---------------------------------------------------------------------

    override suspend fun streamText(
        providerSetting: ProviderSetting.Zhipu,
        messages: List<UIMessage>,
        params: TextGenerationParams,
    ): Flow<StreamChunk> = callbackFlow {
        val requestBody = buildRequestBody(providerSetting, messages, params, stream = true)

        val request = Request.Builder()
            .url(endpoint(providerSetting))
            .headers(params.customHeaders.toHeaders())
            .post(requestBody.toString().toRequestBody(JSON_MEDIA_TYPE))
            .addHeader("Authorization", "Bearer ${authToken(providerSetting)}")
            .addHeader("Content-Type", "application/json")
            .addHeader("Accept", "text/event-stream")
            .build()

        Log.d(TAG, "streamText: $requestBody")

        val decoder = ZhipuStreamDecoder()

        fun sendChunks(chunks: Iterable<StreamChunk>) {
            chunks.forEach { chunk ->
                trySend(chunk).onFailure { e ->
                    Log.w(TAG, "streamText: chunk dropped (${e?.message})")
                }
            }
        }

        val listener = object : EventSourceListener() {
            override fun onEvent(
                eventSource: EventSource,
                id: String?,
                type: String?,
                data: String,
            ) {
                try {
                    val result = decoder.accept(SseEvent(id = id, event = type, data = data))
                    sendChunks(result.chunks)
                    if (result.completed) close()
                } catch (e: Throwable) {
                    close(e)
                }
            }

            override fun onFailure(
                eventSource: EventSource,
                t: Throwable?,
                response: Response?,
            ) {
                val bodyRaw = response.errorBody()
                val exception = when {
                    !bodyRaw.isNullOrBlank() -> runCatching {
                        val element = json.parseToJsonElement(bodyRaw)
                        val obj = element.jsonObjectOrNull
                        if (obj != null && isErrorPayload(obj)) parseNativeError(obj)
                        else element.parseErrorDetail()
                    }.getOrElse { t ?: HttpException(bodyRaw) }

                    else -> t ?: HttpException("Zhipu stream failed: HTTP ${response?.code}")
                }
                Log.w(TAG, "streamText onFailure: ${exception.message}")
                close(exception)
            }

            override fun onClosed(eventSource: EventSource) {
                sendChunks(decoder.onClosed())
                close()
            }
        }

        val eventSource = EventSources.createFactory(client).newEventSource(request, listener)

        awaitClose {
            Log.d(TAG, "streamText: cancel event source")
            eventSource.cancel()
        }
        // trySend 在缓冲满时会静默丢弃 delta, 导致回复中间缺字, 因此缓冲必须无界
    }.buffer(Channel.UNLIMITED).flowOn(Dispatchers.IO)

    // ---------------------------------------------------------------------
    // 原生请求体构造
    // ---------------------------------------------------------------------

    private fun buildRequestBody(
        providerSetting: ProviderSetting.Zhipu,
        messages: List<UIMessage>,
        params: TextGenerationParams,
        stream: Boolean,
    ): JsonObject {
        return buildJsonObject {
            put("model", params.model.modelId)
            put("messages", buildNativeMessages(messages, params.model.inputModalities))
            put("stream", stream)

            // 智谱专有字段, 便于服务端与本地日志对账
            put("request_id", UUID.randomUUID().toString())

            // 智谱 temperature 取值范围 (0.0, 1.0), top_p 取值范围 [0.0, 1.0]
            params.temperature?.let { put("temperature", it.coerceIn(0.01f, 1.0f)) }
            params.topP?.let { put("top_p", it.coerceIn(0.0f, 1.0f)) }
            params.maxTokens?.let { put("max_tokens", it) }

            if (params.model.abilities.contains(ModelAbility.REASONING)) {
                val level = params.reasoningLevel
                putJsonObject("thinking") {
                    put("type", if (level.isEnabled) "enabled" else "disabled")
                }
            }

            if (params.model.abilities.contains(ModelAbility.TOOL) && params.tools.isNotEmpty()) {
                putJsonArray("tools") {
                    params.tools.forEach { tool ->
                        add(buildJsonObject {
                            put("type", "function")
                            put("function", buildJsonObject {
                                put("name", tool.name)
                                put("description", tool.description)
                                put("parameters", tool.parametersSchema())
                            })
                        })
                    }
                }
                put("tool_choice", "auto")
            }
        }.mergeCustomBody(params.customBody)
    }

    private fun buildNativeMessages(
        messages: List<UIMessage>,
        inputModalities: List<Modality>,
    ): JsonArray = buildJsonArray {
        messages.filter { it.isValidToUpload() }.forEach { message ->
            if (message.role == MessageRole.ASSISTANT) {
                addAssistantMessages(message, inputModalities)
            } else {
                add(buildJsonObject {
                    put("role", message.role.name.lowercase())
                    putNativeContent(message.parts, inputModalities)
                })
            }
        }
    }

    /**
     * assistant 消息需要把「正文」与「工具调用 + 工具结果」拆成多条原生消息,
     * 保证每个 tool_call 后面紧跟对应的 tool 结果。
     */
    private fun JsonArrayBuilder.addAssistantMessages(
        message: UIMessage,
        inputModalities: List<Modality>,
    ) {
        for (segment in groupZhipuParts(message.parts)) {
            when (segment) {
                is ZhipuPartSegment.Content -> {
                    val contentParts = segment.parts.filter {
                        it is UIMessagePart.Text || it is UIMessagePart.Image
                    }
                    if (contentParts.isNotEmpty()) {
                        add(buildJsonObject {
                            put("role", "assistant")
                            putNativeContent(contentParts, inputModalities)
                        })
                    }
                }

                is ZhipuPartSegment.Tools -> {
                    add(buildJsonObject {
                        put("role", "assistant")
                        put("content", "")
                        put("tool_calls", buildJsonArray {
                            segment.tools.forEach { tool ->
                                add(buildJsonObject {
                                    put("id", tool.toolCallId)
                                    put("type", "function")
                                    put("function", buildJsonObject {
                                        put("name", tool.toolName)
                                        // 归一化, 避免流式中断产生的残缺 JSON 被原样回传
                                        put("arguments", tool.inputAsJson().toString())
                                    })
                                })
                            }
                        })
                    })

                    segment.tools.forEach { tool ->
                        add(buildJsonObject {
                            put("role", "tool")
                            put("tool_call_id", tool.toolCallId)
                            putNativeContent(tool.output, inputModalities)
                        })
                    }
                }
            }
        }
    }

    /**
     * 智谱原生 content: 纯文本用字符串, 含图片时用
     * `[{"type":"text",...},{"type":"image_url","image_url":{"url":...}}]` 数组。
     */
    private fun JsonObjectBuilder.putNativeContent(
        parts: List<UIMessagePart>,
        inputModalities: List<Modality>,
    ) {
        val contentParts = parts.filter { it is UIMessagePart.Text || it is UIMessagePart.Image }
        val supportsImage = Modality.IMAGE in inputModalities
        val hasImage = contentParts.any { it is UIMessagePart.Image } && supportsImage

        if (!hasImage) {
            put(
                "content",
                contentParts.joinToString("\n") { part ->
                    when (part) {
                        is UIMessagePart.Text -> part.text
                        is UIMessagePart.Image ->
                            "[Image omitted: current model does not support image input]"

                        else -> ""
                    }
                }
            )
            return
        }

        putJsonArray("content") {
            contentParts.forEach { part ->
                when (part) {
                    is UIMessagePart.Text -> add(buildJsonObject {
                        put("type", "text")
                        put("text", part.text)
                    })

                    is UIMessagePart.Image -> part.encodeBase64()
                        .onSuccess { encoded ->
                            add(buildJsonObject {
                                put("type", "image_url")
                                put("image_url", buildJsonObject { put("url", encoded.base64) })
                            })
                        }
                        .onFailure { Log.w(TAG, "encode image failed: ${part.url}", it) }

                    else -> Unit
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // 鉴权
    // ---------------------------------------------------------------------

    /**
     * 生成 `Authorization: Bearer` 后面使用的令牌。
     *
     * 关闭 [ProviderSetting.Zhipu.jwtAuth] 时直接使用平台下发的 API Key;
     * 开启时按智谱原生规范, 用 `{id}.{secret}` 现场签发一枚 HS256 JWT。
     */
    private fun authToken(providerSetting: ProviderSetting.Zhipu): String {
        val apiKey = keyRoulette.next(providerSetting.apiKey, providerSetting.id.toString())
        require(apiKey.isNotBlank()) { "Zhipu API Key is not configured" }
        if (!providerSetting.jwtAuth) return apiKey
        return runCatching { buildZhipuJwt(apiKey) }
            .getOrElse {
                Log.w(TAG, "build jwt failed, fallback to raw api key", it)
                apiKey
            }
    }

    private fun buildZhipuJwt(apiKey: String): String {
        val separator = apiKey.indexOf('.')
        require(separator > 0 && separator < apiKey.length - 1) {
            "Zhipu API Key should be in the format of {id}.{secret}"
        }
        val id = apiKey.substring(0, separator)
        val secret = apiKey.substring(separator + 1)
        val now = System.currentTimeMillis()

        val header = buildJsonObject {
            put("alg", "HS256")
            put("sign_type", "HS256")
        }.toString()

        val payload = buildJsonObject {
            put("api_key", id)
            put("exp", now + JWT_EXPIRE_DURATION_MS)
            put("timestamp", now)
        }.toString()

        val unsigned = "${base64Url(header.toByteArray(Charsets.UTF_8))}." +
            base64Url(payload.toByteArray(Charsets.UTF_8))

        val mac = Mac.getInstance(HMAC_SHA256)
        mac.init(SecretKeySpec(secret.toByteArray(Charsets.UTF_8), HMAC_SHA256))
        val signature = base64Url(mac.doFinal(unsigned.toByteArray(Charsets.UTF_8)))

        return "$unsigned.$signature"
    }

    private fun base64Url(bytes: ByteArray): String =
        Base64.getUrlEncoder().withoutPadding().encodeToString(bytes)

    // ---------------------------------------------------------------------
    // 工具函数
    // ---------------------------------------------------------------------

    private fun endpoint(providerSetting: ProviderSetting.Zhipu): String {
        val base = providerSetting.baseUrl.trimEnd('/')
        val path = providerSetting.chatCompletionsPath.ifBlank { DEFAULT_CHAT_PATH }
        return base + if (path.startsWith("/")) path else "/$path"
    }

    private fun Response?.errorBody(): String? =
        this?.let { response -> runCatching { response.body.stringSafe() }.getOrNull() }

    private fun httpError(code: Int, bodyRaw: String?, prefix: String): Exception {
        if (bodyRaw.isNullOrBlank()) return HttpException("$prefix: HTTP $code")
        return runCatching {
            val element = json.parseToJsonElement(bodyRaw)
            val obj = element.jsonObjectOrNull
            if (obj != null && isErrorPayload(obj)) parseNativeError(obj) else element.parseErrorDetail()
        }.getOrElse { HttpException("$prefix: HTTP $code, $bodyRaw") }
    }

    /**
     * 智谱错误体有两种形态:
     * `{"error":{"code":"...","message":"..."}}` 与 `{"code":"...","message":"...","success":false}`
     */
    private fun isErrorPayload(payload: JsonObject): Boolean {
        if (payload["error"] != null) return true
        if (payload["success"]?.jsonPrimitiveOrNull?.contentOrNull == "false") return true
        return payload["code"] != null && payload["choices"] == null
    }

    private fun parseNativeError(payload: JsonObject): HttpException {
        val error = payload["error"]?.jsonObjectOrNull
        val code = error?.get("code")?.jsonPrimitiveOrNull?.contentOrNull
            ?: payload["code"]?.jsonPrimitiveOrNull?.contentOrNull
        val message = error?.get("message")?.jsonPrimitiveOrNull?.contentOrNull
            ?: payload["message"]?.jsonPrimitiveOrNull?.contentOrNull
        val requestId = payload["request_id"]?.jsonPrimitiveOrNull?.contentOrNull
            ?: error?.get("request_id")?.jsonPrimitiveOrNull?.contentOrNull
        return HttpException(
            buildString {
                if (!code.isNullOrBlank()) append("[$code] ")
                append(message ?: payload.toString())
                if (!requestId.isNullOrBlank()) append(" (request_id: $requestId)")
            }
        )
    }

    private fun parseUsage(usage: JsonObject?): TokenUsage? {
        if (usage == null) return null
        val prompt = usage["prompt_tokens"]?.jsonPrimitive?.intOrNull ?: 0
        val completion = usage["completion_tokens"]?.jsonPrimitive?.intOrNull ?: 0
        val total = usage["total_tokens"]?.jsonPrimitive?.intOrNull ?: (prompt + completion)
        if (prompt == 0 && completion == 0 && total == 0) return null
        val cached = usage["prompt_cache_hit_tokens"]?.jsonPrimitive?.intOrNull
            ?: usage["cached_tokens"]?.jsonPrimitive?.intOrNull
            ?: 0
        return TokenUsage(
            promptTokens = prompt,
            completionTokens = completion,
            totalTokens = total,
            cachedTokens = cached,
        )
    }

    private fun parseNativeMessage(messageJson: JsonObject): UIMessage {
        val role = runCatching {
            MessageRole.valueOf(
                messageJson["role"]?.jsonPrimitiveOrNull?.contentOrNull?.uppercase() ?: "ASSISTANT"
            )
        }.getOrDefault(MessageRole.ASSISTANT)

        val reasoning = messageJson["reasoning_content"]?.jsonPrimitiveOrNull?.contentOrNull
        val contentElement = messageJson["content"]
        val text = contentElement?.jsonPrimitiveOrNull?.contentOrNull
            ?: contentElement?.jsonArrayOrNull?.mapNotNull { item ->
                item.jsonObjectOrNull?.get("text")?.jsonPrimitiveOrNull?.contentOrNull
            }?.joinToString("").orEmpty()

        return UIMessage(
            role = role,
            parts = buildList {
                if (!reasoning.isNullOrEmpty()) {
                    add(UIMessagePart.Reasoning(reasoning = reasoning, finishedAt = null))
                }
                messageJson["tool_calls"]?.jsonArrayOrNull?.forEach { element ->
                    val toolCall = element.jsonObjectOrNull ?: return@forEach
                    val function = toolCall["function"]?.jsonObjectOrNull
                    add(
                        UIMessagePart.Tool(
                            toolCallId = toolCall["id"]?.jsonPrimitiveOrNull?.contentOrNull.orEmpty(),
                            toolName = function?.get("name")?.jsonPrimitiveOrNull?.contentOrNull.orEmpty(),
                            input = function?.get("arguments")?.jsonPrimitiveOrNull?.contentOrNull.orEmpty(),
                            output = emptyList(),
                        )
                    )
                }
                if (text.isNotEmpty()) add(UIMessagePart.Text(text))
            },
        )
    }

    private fun Tool.parametersSchema(): JsonObject {
        val schema = parameters()
        return if (schema is InputSchema.Obj) {
            buildJsonObject {
                put("type", "object")
                put("properties", schema.properties)
                if (!schema.required.isNullOrEmpty()) {
                    put("required", JsonArray(schema.required.orEmpty().map { JsonPrimitive(it) }))
                }
            }
        } else {
            buildJsonObject {
                put("type", "object")
                put("properties", JsonObject(emptyMap()))
            }
        }
    }

    private companion object {
        /** 远端模型列表不可用时使用的常用 GLM 模型兜底 */
        val FALLBACK_MODELS = listOf(
            "glm-4-plus",
            "glm-4-air",
            "glm-4-airx",
            "glm-4-flash",
            "glm-4-long",
            "glm-4.6",
            "glm-4.5v",
        )
    }
}

// -------------------------------------------------------------------------
// 原生 SSE 解码
// -------------------------------------------------------------------------

/** 按工具边界切分消息 parts, 保证 tool_call 与 tool 结果成对出现 */
private sealed class ZhipuPartSegment {
    data class Content(val parts: List<UIMessagePart>) : ZhipuPartSegment()
    data class Tools(val tools: List<UIMessagePart.Tool>) : ZhipuPartSegment()
}

private fun groupZhipuParts(parts: List<UIMessagePart>): List<ZhipuPartSegment> {
    val groups = mutableListOf<ZhipuPartSegment>()
    val content = mutableListOf<UIMessagePart>()
    val tools = mutableListOf<UIMessagePart.Tool>()

    fun flushContent() {
        if (content.isNotEmpty()) {
            groups.add(ZhipuPartSegment.Content(content.toList()))
            content.clear()
        }
    }

    fun flushTools() {
        if (tools.isNotEmpty()) {
            groups.add(ZhipuPartSegment.Tools(tools.toList()))
            tools.clear()
        }
    }

    for (part in parts) {
        // 只回传已执行的工具; 待审批的调用与内置适配器保持一致, 直接跳过
        if (part is UIMessagePart.Tool && part.isExecuted) {
            flushContent()
            tools.add(part)
        } else {
            flushTools()
            if (part !is UIMessagePart.Tool) content.add(part)
        }
    }

    flushContent()
    flushTools()
    return groups
}

/**
 * 智谱原生 SSE 解码器。
 *
 * 智谱的流式增量放在 `choices[0].delta` 中, 思考内容走 `reasoning_content`,
 * 以 `data: [DONE]` 或 `finish_reason` 非空作为结束标志。每条响应流必须使用独立实例。
 */
private class ZhipuStreamDecoder : StreamChunkDecoder {

    private val state = ZhipuStreamState()
    private val toolIdsByIndex = mutableMapOf<Int, String>()

    private var responseId: String? = null
    private var responseModel: String? = null
    private var finishReason: String? = null
    private var finished = false

    override fun accept(event: SseEvent): DecodeResult {
        if (finished) return DecodeResult(completed = true)

        val raw = event.data.trim()
        if (raw.isEmpty()) return DecodeResult()
        if (raw == "[DONE]") return DecodeResult(finish(), completed = true)

        val payload = runCatching { json.parseToJsonElement(raw).jsonObjectOrNull }.getOrNull()
            ?: return DecodeResult()

        if (isErrorPayload(payload)) throw parseNativeError(payload)

        val chunks = mutableListOf<StreamChunk>()
        responseId = payload["id"]?.jsonPrimitiveOrNull?.contentOrNull
            ?: payload["request_id"]?.jsonPrimitiveOrNull?.contentOrNull
            ?: responseId
        responseModel = payload["model"]?.jsonPrimitiveOrNull?.contentOrNull ?: responseModel

        val choice = payload["choices"]?.jsonArrayOrNull?.firstOrNull()?.jsonObjectOrNull
        val delta = choice?.get("delta")?.jsonObjectOrNull ?: choice?.get("message")?.jsonObjectOrNull

        if (delta != null) {
            val reasoning = delta["reasoning_content"]?.jsonPrimitiveOrNull?.contentOrNull.orEmpty()
            if (reasoning.isNotEmpty()) chunks += state.reasoning(responseId, reasoning)

            val text = delta["content"]?.jsonPrimitiveOrNull?.contentOrNull.orEmpty()
            if (text.isNotEmpty()) chunks += state.text(responseId, text)

            delta["tool_calls"]?.jsonArrayOrNull?.forEachIndexed { fallbackIndex, element ->
                val toolCall = element.jsonObjectOrNull ?: return@forEachIndexed
                val index = toolCall["index"]?.jsonPrimitive?.intOrNull ?: fallbackIndex
                val toolId = toolCall["id"]?.jsonPrimitiveOrNull?.contentOrNull
                    ?.also { toolIdsByIndex[index] = it }
                    ?: toolIdsByIndex.getOrPut(index) { "${responseId ?: "response"}:tool-$index" }
                val function = toolCall["function"]?.jsonObjectOrNull
                chunks += state.tool(
                    sourceId = responseId,
                    toolCallId = toolId,
                    toolName = function?.get("name")?.jsonPrimitiveOrNull?.contentOrNull.orEmpty(),
                    inputDelta = function?.get("arguments")?.jsonPrimitiveOrNull?.contentOrNull.orEmpty(),
                )
            }

            finishReason = choice?.get("finish_reason")?.jsonPrimitiveOrNull?.contentOrNull
                ?: finishReason
        }

        parseUsage(payload["usage"]?.jsonObjectOrNull)?.let { chunks += StreamChunk.Usage(it) }

        return if (finishReason != null) {
            chunks += finish()
            DecodeResult(chunks, completed = true)
        } else {
            DecodeResult(chunks)
        }
    }

    override fun onClosed(): List<StreamChunk> = finish()

    private fun finish(): List<StreamChunk> {
        if (finished) return emptyList()
        finished = true
        return state.finish(finishReason, responseId, responseModel)
    }

    private fun parseUsage(usage: JsonObject?): TokenUsage? {
        if (usage == null) return null
        val prompt = usage["prompt_tokens"]?.jsonPrimitive?.intOrNull ?: 0
        val completion = usage["completion_tokens"]?.jsonPrimitive?.intOrNull ?: 0
        val total = usage["total_tokens"]?.jsonPrimitive?.intOrNull ?: (prompt + completion)
        if (prompt == 0 && completion == 0 && total == 0) return null
        val cached = usage["prompt_cache_hit_tokens"]?.jsonPrimitive?.intOrNull
            ?: usage["cached_tokens"]?.jsonPrimitive?.intOrNull
            ?: 0
        return TokenUsage(
            promptTokens = prompt,
            completionTokens = completion,
            totalTokens = total,
            cachedTokens = cached,
        )
    }

    private fun isErrorPayload(payload: JsonObject): Boolean {
        if (payload["error"] != null) return true
        if (payload["success"]?.jsonPrimitiveOrNull?.contentOrNull == "false") return true
        return payload["code"] != null && payload["choices"] == null
    }

    private fun parseNativeError(payload: JsonObject): HttpException {
        val error = payload["error"]?.jsonObjectOrNull
        val code = error?.get("code")?.jsonPrimitiveOrNull?.contentOrNull
            ?: payload["code"]?.jsonPrimitiveOrNull?.contentOrNull
        val message = error?.get("message")?.jsonPrimitiveOrNull?.contentOrNull
            ?: payload["message"]?.jsonPrimitiveOrNull?.contentOrNull
        val requestId = payload["request_id"]?.jsonPrimitiveOrNull?.contentOrNull
        return HttpException(
            buildString {
                if (!code.isNullOrBlank()) append("[$code] ")
                append(message ?: payload.toString())
                if (!requestId.isNullOrBlank()) append(" (request_id: $requestId)")
            }
        )
    }
}

/** 把原生增量事件翻译成 Provider 无关的 [StreamChunk] 序列 */
private class ZhipuStreamState {
    private var sequence = 0
    private var textId: String? = null
    private var reasoningId: String? = null
    private val openToolIds = linkedSetOf<String>()
    private var lastToolId: String? = null

    fun text(sourceId: String?, delta: String): List<StreamChunk> {
        if (delta.isEmpty()) return emptyList()
        return buildList {
            addAll(closeReasoning())
            addAll(closeTools())
            val id = textId ?: nextId(sourceId, "text").also {
                textId = it
                add(StreamChunk.TextStart(it))
            }
            add(StreamChunk.TextDelta(id, delta))
        }
    }

    fun reasoning(sourceId: String?, delta: String): List<StreamChunk> {
        if (delta.isEmpty()) return emptyList()
        return buildList {
            addAll(closeText())
            addAll(closeTools())
            val id = reasoningId ?: nextId(sourceId, "reasoning").also {
                reasoningId = it
                add(StreamChunk.ReasoningStart(it))
            }
            add(StreamChunk.ReasoningDelta(id, delta))
        }
    }

    fun tool(
        sourceId: String?,
        toolCallId: String,
        toolName: String,
        inputDelta: String,
    ): List<StreamChunk> = buildList {
        addAll(closeText())
        addAll(closeReasoning())
        val id = toolCallId.ifBlank { lastToolId ?: nextId(sourceId, "tool") }
        var nameDelta = toolName
        if (openToolIds.add(id)) {
            nameDelta = ""
            add(StreamChunk.ToolCallStart(id, toolName))
        }
        lastToolId = id
        if (nameDelta.isNotEmpty() || inputDelta.isNotEmpty()) {
            add(StreamChunk.ToolCallDelta(id, nameDelta, inputDelta))
        }
    }

    fun finish(reason: String?, responseId: String?, model: String?): List<StreamChunk> = buildList {
        addAll(closeText())
        addAll(closeReasoning())
        addAll(closeTools())
        add(StreamChunk.Finish(reason, responseId, model))
    }

    private fun closeText() =
        textId?.let { textId = null; listOf(StreamChunk.TextEnd(it)) }.orEmpty()

    private fun closeReasoning() =
        reasoningId?.let { reasoningId = null; listOf(StreamChunk.ReasoningEnd(it)) }.orEmpty()

    private fun closeTools() = openToolIds.toList().map { StreamChunk.ToolCallEnd(it) }.also {
        openToolIds.clear()
        lastToolId = null
    }

    private fun nextId(sourceId: String?, kind: String): String =
        "${sourceId?.takeIf(String::isNotBlank)?.let { "$it:" }.orEmpty()}$kind-${++sequence}"
}
