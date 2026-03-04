package com.transbloom.cameraapp.inference

import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.util.concurrent.atomic.AtomicBoolean

class StillFrameAnalyzer(
    private val classifier: FlowerOnnxClassifier,
    private val onStabilityUpdate: (StabilityState) -> Unit,
    private val onPrediction: (Prediction) -> Unit,
    private val cooldownMs: Long = 1500L,
) : ImageAnalysis.Analyzer {
    private val stabilityDetector = FrameStabilityDetector()
    private val isInferring = AtomicBoolean(false)
    private var lastInferenceTimeMs: Long = 0L

    override fun analyze(image: ImageProxy) {
        try {
            val plane = image.planes.firstOrNull() ?: return
            val buffer = plane.buffer
            val rgba = ByteArray(buffer.remaining())
            buffer.get(rgba)

            val state = stabilityDetector.update(rgba, image.width, image.height)
            onStabilityUpdate(state)

            val now = System.currentTimeMillis()
            val canInfer = state.isStable && (now - lastInferenceTimeMs) >= cooldownMs
            if (!canInfer || !isInferring.compareAndSet(false, true)) {
                return
            }

            try {
                val input = ImageTensorPreprocessor.rgbaToNchwFloat32(
                    rgba = rgba,
                    srcWidth = image.width,
                    srcHeight = image.height,
                    dstSize = 32,
                )
                val prediction = classifier.predict(input)
                lastInferenceTimeMs = now
                onPrediction(prediction)
            } finally {
                isInferring.set(false)
            }
        } finally {
            image.close()
        }
    }
}
