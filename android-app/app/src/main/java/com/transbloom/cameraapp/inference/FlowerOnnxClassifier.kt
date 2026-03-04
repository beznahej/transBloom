package com.transbloom.cameraapp.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer
import org.json.JSONObject

data class Prediction(
    val index: Int,
    val label: String,
    val confidence: Float,
    val probabilities: FloatArray,
)

class FlowerOnnxClassifier(
    context: Context,
    modelAssetName: String = "flowernet.onnx",
    classMapAssetName: String = "class_to_idx.json",
) : AutoCloseable {
    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val classNames: List<String>

    init {
        val modelBytes = context.assets.open(modelAssetName).use { it.readBytes() }
        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(1)
            setInterOpNumThreads(1)
        }
        session = environment.createSession(modelBytes, sessionOptions)
        inputName = session.inputNames.first()
        classNames = loadClassNames(context, classMapAssetName)
    }

    fun predict(input: FloatArray): Prediction {
        val inputShape = longArrayOf(1, 3, 32, 32)
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(input), inputShape).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { result ->
                val raw = result[0].value
                val logits = when (raw) {
                    is Array<*> -> raw.firstOrNull() as? FloatArray
                    is FloatArray -> raw
                    else -> null
                } ?: throw IllegalStateException("Unexpected ONNX output type: ${raw?.javaClass}")

                val probabilities = softmax(logits)
                val index = argmax(probabilities)
                val label = classNames.getOrElse(index) { index.toString() }
                return Prediction(
                    index = index,
                    label = label,
                    confidence = probabilities[index],
                    probabilities = probabilities,
                )
            }
        }
    }

    private fun loadClassNames(context: Context, classMapAssetName: String): List<String> {
        val fallback = listOf("flower", "non_flower")
        return runCatching {
            val text = context.assets.open(classMapAssetName).bufferedReader().use { it.readText() }
            val json = JSONObject(text)
            val pairs = mutableListOf<Pair<Int, String>>()
            val keys = json.keys()
            while (keys.hasNext()) {
                val className = keys.next()
                val index = json.getInt(className)
                pairs += index to className
            }
            pairs.sortedBy { it.first }.map { it.second }
        }.getOrElse { fallback }
    }

    override fun close() {
        session.close()
    }

    private fun softmax(values: FloatArray): FloatArray {
        val max = values.maxOrNull() ?: 0f
        val exps = FloatArray(values.size)
        var sum = 0f
        for (i in values.indices) {
            val exp = kotlin.math.exp((values[i] - max).toDouble()).toFloat()
            exps[i] = exp
            sum += exp
        }
        if (sum == 0f) return FloatArray(values.size)
        for (i in exps.indices) {
            exps[i] = exps[i] / sum
        }
        return exps
    }

    private fun argmax(values: FloatArray): Int {
        var bestIdx = 0
        var bestVal = values[0]
        for (i in 1 until values.size) {
            if (values[i] > bestVal) {
                bestVal = values[i]
                bestIdx = i
            }
        }
        return bestIdx
    }
}
