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

private const val TAG = "BailianNativeProvider"

private val json = Json {
    ignoreUnknownKeys = true
    explicitNulls = false
}

private val JSON_MEDIA_TYPE: MediaType = "application/json".toMediaType()

private const val DEFAULT_GENERATION_PATH = "/services/aigc/text-generation/generation"

/**
 * 阿里云百炼 (DashScope) **原生协议** ChatProvider 实现。
 *
 * 与项目内置的 OpenAI 兼容适配器完全独立, 不复用其请求构造与流式解码逻辑:
 * - 请求体使用 DashScope 原生的 `model` / `input` / `parameters` 三段式结构;
 * - 流式通过 `X-DashScope-SSE: enable` 请求头开启, 事件形如 `event:result` / `event:error`;
 * - 响应从 `output.choices[0].message` 取值, `finish_reason` 在进行中时是字符串 `"null"`;
 * - 鉴权使用 `Authorization: Bearer {API-KEY}`, 支持多 Key 轮询。
 */
class BailianNativeProvider(
    private val client: OkHttpClient,
    context: Context? = null,
) : Provider<ProviderSetting.Bailian> {

    private val keyRoulette: KeyRoulette =
        if (context != null) KeyRoulette.lru(context) else KeyRoulette.default()

    // ---------------------------------------------------------------------
    // 模型列表
    // ---------------------------------------------------------------------

    override suspend fun listModels(providerSetting: ProviderSetting.Bailian): List<Model> =
        withContext(Dispatchers.IO) {
            val remote = runCatching { fetchRemoteModels(providerSetting) }
                .onFailure { Log.w(TAG, "listModels failed, fallback to built-in list", it) }
                .getOrDefault(emptyList())
            remote.ifEmpty { FALLBACK_MODELS.map { Model(modelId = it, displayName = it) } }
        }

    private suspend fun fetchRemoteModels(providerSetting: ProviderSetting.Bailian): List<Model> {
        val request = Request.Builder()
            .url("${providerSetting.baseUrl.trimEnd('/')}/models")
            .addHeader("Authorization", "Bearer ${nextKey(providerSetting)}")
            .get()
            .build()

        val response = client.newCall(request).await()
        val bodyStr = response.body.string()
        if (!response.isSuccessful) {
            throw httpError(response.code, bodyStr, "Failed to get Bailian models")
        }

        val payload = json.parseToJsonElement(bodyStr).jsonObjectOrNull ?: return emptyList()
        val array = payload["data"]?.jsonArrayOrNull
            ?: payload["output"]?.jsonObjectOrNull?.get("models")?.jsonArrayOrNull
            ?: payload["models"]?.jsonArrayOrNull
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
        providerSetting: ProviderSetting.Bailian,
        messages: List<UIMessage>,
        params: TextGenerationParams,
    ): TextGenerationResult = withContext(Dispatchers.IO) {
        val requestBody = buildRequestBody(providerSetting, messages, params, stream = false)

        val request = Request.Builder()
            .url(endpoint(providerSetting))
            .headers(params.customHeaders.toHeaders())
            .post(requestBody.toString().toRequestBody(JSON_MEDIA_TYPE))
            .addHeader("Authorization", "Bearer ${nextKey(providerSetting)}")
            .addHeader("Content-Type", "application/json")
            .build()

        val response = client.newCall(request).await()
        val bodyStr = response.body.string()
        if (!response.isSuccessful) {
            throw httpError(response.code, bodyStr, "Bailian generateText failed")
        }

        val payload = runCatching { json.parseToJsonElement(bodyStr).jsonObject }
            .getOrElse { throw HttpException("Invalid Bailian response: $bodyStr") }

        val output = payload["output"]?.jsonObjectOrNull ?: throw parseNativeError(payload)
        val requestId = payload["request_id"]?.jsonPrimitiveOrNull?.contentOrNull.orEmpty()
        val choice = output["choices"]?.jsonArrayOrNull?.firstOrNull()?.jsonObjectOrNull
        val messageJson = choice?.get("message")?.jsonObjectOrNull

        val message = if (messageJson != null) {
            parseNativeMessage(messageJson)
        } else {
            // result_format=text 的兜底
            UIMessage(
                role = MessageRole.ASSISTANT,
                parts = listOf(
                    UIMessagePart.Text(output["text"]?.jsonPrimitiveOrNull?.contentOrNull.orEmpty())
                ),
            )
        }

        val finishReason = normalizeFinishReason(
            (choice?.get("finish_reason") ?: output["finish_reason"])
                ?.jsonPrimitiveOrNull?.contentOrNull
        )

        TextGenerationResult(
            id = requestId,
            model = payload["model"]?.jsonPrimitiveOrNull?.contentOrNull ?: params.model.modelId,
            message = message,
            finishReason = finishReason,
            usage = parseUsage(payload["usage"]?.jsonObjectOrNull),
        )
    }

    // ---------------------------------------------------------------------
    // 流式生成 (DashScope 原生 SSE)
    // ---------------------------------------------------------------------

    override suspend fun streamText(
        providerSetting: ProviderSetting.Bailian,
        messages: List<UIMessage>,
        params: TextGenerationParams,
    ): Flow<StreamChunk> = callbackFlow {
        val requestBody = buildRequestBody(providerSetting, messages, params, stream = true)

        val request = Request.Builder()
            .url(endpoint(providerSetting))
            .headers(params.customHeaders.toHeaders())
            .post(requestBody.toString().toRequestBody(JSON_MEDIA_TYPE))
            .addHeader("Authorization", "Bearer ${nextKey(providerSetting)}")
            .addHeader("Content-Type", "application/json")
            .addHeader("Accept", "text/event-stream")
            // DashScope 原生 SSE 开关, 缺少该请求头服务端会退化成一次性返回
            .addHeader("X-DashScope-SSE", "enable")
            .build()

        Log.d(TAG, "streamText: $requestBody")

        val decoder = BailianStreamDecoder(providerSetting.incrementalOutput)

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
                        json.parseToJsonElement(bodyRaw).parseErrorDetail()
                    }.getOrElse { t ?: HttpException(bodyRaw) }

                    else -> t ?: HttpException("Bailian stream failed: HTTP ${response?.code}")
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
        providerSetting: ProviderSetting.Bailian,
        messages: List<UIMessage>,
        params: TextGenerationParams,
        stream: Boolean,
    ): JsonObject {
        require(providerSetting.apiKey.isNotBlank()) { "Bailian API Key is not configured" }

        return buildJsonObject {
            put("model", params.model.modelId)

            putJsonObject("input") {
                put("messages", buildNativeMessages(messages, params.model.inputModalities))
            }

            putJsonObject("parameters") {
                // 必须显式指定, 否则服务端默认回传 result_format=text
                put("result_format", "message")

                if (stream) {
                    put("incremental_output", providerSetting.incrementalOutput)
                }

                params.temperature?.let { put("temperature", it) }
                params.topP?.let { put("top_p", it) }
                params.maxTokens?.let { put("max_tokens", it) }

                if (params.model.abilities.contains(ModelAbility.REASONING)) {
                    val level = params.reasoningLevel
                    // DashScope 的深度思考只支持流式输出, 非流式开启会被服务端直接拒绝
                    val enableThinking = level.isEnabled && stream
                    put("enable_thinking", enableThinking)
                    if (enableThinking && level != ReasoningLevel.AUTO) {
                        put("thinking_budget", level.budgetTokens)
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
                }
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
        for (segment in groupPartsByToolBoundary(message.parts)) {
            when (segment) {
                is PartSegment.Content -> {
                    val reasoning = segment.parts
                        .filterIsInstance<UIMessagePart.Reasoning>()
                        .joinToString("") { it.reasoning }
                    val contentParts = segment.parts.filter {
                        it is UIMessagePart.Text || it is UIMessagePart.Image
                    }
                    if (contentParts.isNotEmpty() || reasoning.isNotBlank()) {
                        add(buildJsonObject {
                            put("role", "assistant")
                            if (reasoning.isNotBlank()) put("reasoning_content", reasoning)
                            putNativeContent(contentParts, inputModalities)
                        })
                    }
                }

                is PartSegment.Tools -> {
                    add(buildJsonObject {
                        put("role", "assistant")
                        put("content", "")
                        put("tool_calls", buildJsonArray {
                            segment.tools.forEachIndexed { index, tool ->
                                add(buildJsonObject {
                                    put("index", index)
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
                            put("name", tool.toolName)
                            putNativeContent(tool.output, inputModalities)
                        })
                    }
                }
            }
        }
    }

    /**
     * DashScope 原生 content: 纯文本用字符串, 含图片时用 `[{"text":...},{"image":...}]` 数组。
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
                    is UIMessagePart.Text -> add(buildJsonObject { put("text", part.text) })
                    is UIMessagePart.Image -> part.encodeBase64()
                        .onSuccess { encoded -> add(buildJsonObject { put("image", encoded.base64) }) }
                        .onFailure { Log.w(TAG, "encode image failed: ${part.url}", it) }

                    else -> Unit
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // 工具函数
    // ---------------------------------------------------------------------

    private fun endpoint(providerSetting: ProviderSetting.Bailian): String {
        val base = providerSetting.baseUrl.trimEnd('/')
        val path = providerSetting.generationPath.ifBlank { DEFAULT_GENERATION_PATH }
        return base + if (path.startsWith("/")) path else "/$path"
    }

    private fun nextKey(providerSetting: ProviderSetting.Bailian): String {
        require(providerSetting.apiKey.isNotBlank()) { "Bailian API Key is not configured" }
        return keyRoulette.next(providerSetting.apiKey, providerSetting.id.toString())
    }

    private fun Response?.errorBody(): String? =
        this?.let { response -> runCatching { response.body.stringSafe() }.getOrNull() }

    private fun httpError(code: Int, bodyRaw: String?, prefix: String): Exception {
        if (bodyRaw.isNullOrBlank()) return HttpException("$prefix: HTTP $code")
        return runCatching {
            val element = json.parseToJsonElement(bodyRaw)
            val obj = element.jsonObjectOrNull
            if (obj != null && (obj["code"] != null || obj["message"] != null)) {
                parseNativeError(obj)
            } else {
                element.parseErrorDetail()
            }
        }.getOrElse { HttpException("$prefix: HTTP $code, $bodyRaw") }
    }

    /** DashScope 错误体: `{"code":"InvalidApiKey","message":"...","request_id":"..."}` */
    private fun parseNativeError(payload: JsonObject): HttpException {
        val code = payload["code"]?.jsonPrimitiveOrNull?.contentOrNull
        val message = payload["message"]?.jsonPrimitiveOrNull?.contentOrNull
            ?: payload["error"]?.jsonObjectOrNull?.get("message")?.jsonPrimitiveOrNull?.contentOrNull
        val requestId = payload["request_id"]?.jsonPrimitiveOrNull?.contentOrNull
        return HttpException(
            buildString {
                if (!code.isNullOrBlank()) append("[$code] ")
                append(message ?: payload.toString())
                if (!requestId.isNullOrBlank()) append(" (request_id: $requestId)")
            }
        )
    }

    /** 流式进行中 DashScope 会把 finish_reason 置为字符串 "null" */
    private fun normalizeFinishReason(raw: String?): String? {
        if (raw.isNullOrBlank()) return null
        return when (raw.lowercase()) {
            "null", "none" -> null
            else -> raw
        }
    }

    private fun parseUsage(usage: JsonObject?): TokenUsage? {
        if (usage == null) return null
        val prompt = usage["input_tokens"]?.jsonPrimitive?.intOrNull
            ?: usage["prompt_tokens"]?.jsonPrimitive?.intOrNull
            ?: 0
        val completion = usage["output_tokens"]?.jsonPrimitive?.intOrNull
            ?: usage["completion_tokens"]?.jsonPrimitive?.intOrNull
            ?: 0
        val total = usage["total_tokens"]?.jsonPrimitive?.intOrNull ?: (prompt + completion)
        if (prompt == 0 && completion == 0 && total == 0) return null
        val cached = usage["prompt_tokens_details"]?.jsonObjectOrNull
            ?.get("cached_tokens")?.jsonPrimitive?.intOrNull
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
        /** 远端模型列表不可用时使用的常用通义千问模型兜底 */
        val FALLBACK_MODELS = listOf(
            "qwen-max",
            "qwen-plus",
            "qwen-turbo",
            "qwen-long",
            "qwen-vl-max",
            "qwen-vl-plus",
            "qwen-coder-plus",
            "qwq-plus",
        )
    }
}

// -------------------------------------------------------------------------
// 原生 SSE 解码
// -------------------------------------------------------------------------

/** 按工具边界切分消息 parts, 保证 tool_call 与 tool 结果成对出现 */
private sealed class PartSegment {
    data class Content(val parts: List<UIMessagePart>) : PartSegment()
    data class Tools(val tools: List<UIMessagePart.Tool>) : PartSegment()
}

private fun groupPartsByToolBoundary(parts: List<UIMessagePart>): List<PartSegment> {
    val groups = mutableListOf<PartSegment>()
    val content = mutableListOf<UIMessagePart>()
    val tools = mutableListOf<UIMessagePart.Tool>()

    fun flushContent() {
        if (content.isNotEmpty()) {
            groups.add(PartSegment.Content(content.toList()))
            content.clear()
        }
    }

    fun flushTools() {
        if (tools.isNotEmpty()) {
            groups.add(PartSegment.Tools(tools.toList()))
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
 * DashScope 原生 SSE 解码器。
 *
 * 每条响应流必须使用独立实例。[incrementalOutput] 为 false 时服务端回传全量文本,
 * 需要在这里做前缀差分, 转换成上层期望的增量事件。
 */
private class BailianStreamDecoder(
    private val incrementalOutput: Boolean,
) : StreamChunkDecoder {

    private val state = BailianStreamState()
    private val toolIdsByIndex = mutableMapOf<Int, String>()

    private var requestId: String? = null
    private var model: String? = null
    private var finishReason: String? = null
    private var finished = false

    private var accumulatedText = ""
    private var accumulatedReasoning = ""

    override fun accept(event: SseEvent): DecodeResult {
        if (finished) return DecodeResult(completed = true)

        val raw = event.data.trim()
        if (raw.isEmpty()) return DecodeResult()
        if (raw == "[DONE]") return DecodeResult(finish(), completed = true)

        val payload = runCatching { json.parseToJsonElement(raw).jsonObjectOrNull }.getOrNull()
            ?: return DecodeResult()

        // DashScope 出错时会走 event:error, 或在 data 里带 code 而没有 output
        if (event.event == "error" || (payload["output"] == null && payload["code"] != null)) {
            throw parseNativeError(payload)
        }

        val chunks = mutableListOf<StreamChunk>()
        requestId = payload["request_id"]?.jsonPrimitiveOrNull?.contentOrNull ?: requestId
        model = payload["model"]?.jsonPrimitiveOrNull?.contentOrNull ?: model

        val output = payload["output"]?.jsonObjectOrNull
        val choice = output?.get("choices")?.jsonArrayOrNull?.firstOrNull()?.jsonObjectOrNull
        val message = choice?.get("message")?.jsonObjectOrNull

        if (message != null) {
            val reasoning = message["reasoning_content"]?.jsonPrimitiveOrNull?.contentOrNull.orEmpty()
            if (reasoning.isNotEmpty()) {
                val reasoningDelta = diff(reasoning, accumulatedReasoning)
                accumulatedReasoning += reasoningDelta
                chunks += state.reasoning(requestId, reasoningDelta)
            }

            val text = message["content"]?.jsonPrimitiveOrNull?.contentOrNull.orEmpty()
            if (text.isNotEmpty()) {
                val textDelta = diff(text, accumulatedText)
                accumulatedText += textDelta
                chunks += state.text(requestId, textDelta)
            }

            message["tool_calls"]?.jsonArrayOrNull?.forEachIndexed { fallbackIndex, element ->
                val toolCall = element.jsonObjectOrNull ?: return@forEachIndexed
                val index = toolCall["index"]?.jsonPrimitive?.intOrNull ?: fallbackIndex
                val toolId = toolCall["id"]?.jsonPrimitiveOrNull?.contentOrNull
                    ?.also { toolIdsByIndex[index] = it }
                    ?: toolIdsByIndex.getOrPut(index) { "${requestId ?: "response"}:tool-$index" }
                val function = toolCall["function"]?.jsonObjectOrNull
                chunks += state.tool(
                    sourceId = requestId,
                    toolCallId = toolId,
                    toolName = function?.get("name")?.jsonPrimitiveOrNull?.contentOrNull.orEmpty(),
                    inputDelta = function?.get("arguments")?.jsonPrimitiveOrNull?.contentOrNull.orEmpty(),
                )
            }

            finishReason = normalizeFinishReason(
                choice?.get("finish_reason")?.jsonPrimitiveOrNull?.contentOrNull
            ) ?: finishReason
        } else if (output != null) {
            val text = output["text"]?.jsonPrimitiveOrNull?.contentOrNull.orEmpty()
            if (text.isNotEmpty()) {
                val textDelta = diff(text, accumulatedText)
                accumulatedText += textDelta
                chunks += state.text(requestId, textDelta)
            }
            finishReason = normalizeFinishReason(
                output["finish_reason"]?.jsonPrimitiveOrNull?.contentOrNull
            ) ?: finishReason
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
        return state.finish(finishReason, requestId, model)
    }

    /** 非增量模式下把全量文本差分成增量 */
    private fun diff(incoming: String, accumulated: String): String {
        if (incrementalOutput || accumulated.isEmpty()) return incoming
        return if (incoming.startsWith(accumulated)) incoming.removePrefix(accumulated) else incoming
    }

    private fun normalizeFinishReason(raw: String?): String? {
        if (raw.isNullOrBlank()) return null
        return when (raw.lowercase()) {
            "null", "none" -> null
            else -> raw
        }
    }

    private fun parseUsage(usage: JsonObject?): TokenUsage? {
        if (usage == null) return null
        val prompt = usage["input_tokens"]?.jsonPrimitive?.intOrNull
            ?: usage["prompt_tokens"]?.jsonPrimitive?.intOrNull
            ?: 0
        val completion = usage["output_tokens"]?.jsonPrimitive?.intOrNull
            ?: usage["completion_tokens"]?.jsonPrimitive?.intOrNull
            ?: 0
        val total = usage["total_tokens"]?.jsonPrimitive?.intOrNull ?: (prompt + completion)
        if (prompt == 0 && completion == 0 && total == 0) return null
        val cached = usage["prompt_tokens_details"]?.jsonObjectOrNull
            ?.get("cached_tokens")?.jsonPrimitive?.intOrNull
            ?: usage["cached_tokens"]?.jsonPrimitive?.intOrNull
            ?: 0
        return TokenUsage(
            promptTokens = prompt,
            completionTokens = completion,
            totalTokens = total,
            cachedTokens = cached,
        )
    }

    private fun parseNativeError(payload: JsonObject): HttpException {
        val code = payload["code"]?.jsonPrimitiveOrNull?.contentOrNull
        val message = payload["message"]?.jsonPrimitiveOrNull?.contentOrNull
            ?: payload["error"]?.jsonObjectOrNull?.get("message")?.jsonPrimitiveOrNull?.contentOrNull
        val id = payload["request_id"]?.jsonPrimitiveOrNull?.contentOrNull
        return HttpException(
            buildString {
                if (!code.isNullOrBlank()) append("[$code] ")
                append(message ?: payload.toString())
                if (!id.isNullOrBlank()) append(" (request_id: $id)")
            }
        )
    }
}

/** 把原生增量事件翻译成 Provider 无关的 [StreamChunk] 序列 */
private class BailianStreamState {
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
