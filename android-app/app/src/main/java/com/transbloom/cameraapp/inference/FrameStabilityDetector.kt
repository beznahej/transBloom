package com.transbloom.cameraapp.inference

data class StabilityState(
    val motionScore: Float,
    val stableFrames: Int,
    val isStable: Boolean,
)

class FrameStabilityDetector(
    private val motionThreshold: Float = 4.0f,
    private val stableFramesRequired: Int = 8,
    private val sampleStep: Int = 12,
) {
    private var previousLuma: ByteArray? = null
    private var stableFrameCount: Int = 0

    fun update(rgba: ByteArray, width: Int, height: Int): StabilityState {
        val current = sampleLuma(rgba, width, height)

        val motion = if (previousLuma == null) {
            Float.MAX_VALUE
        } else {
            meanAbsoluteDifference(previousLuma!!, current)
        }

        if (motion < motionThreshold) {
            stableFrameCount += 1
        } else {
            stableFrameCount = 0
        }

        previousLuma = current
        val stable = stableFrameCount >= stableFramesRequired
        return StabilityState(
            motionScore = motion,
            stableFrames = stableFrameCount,
            isStable = stable,
        )
    }

    private fun sampleLuma(rgba: ByteArray, width: Int, height: Int): ByteArray {
        val values = ArrayList<Byte>()
        var y = 0
        while (y < height) {
            var x = 0
            while (x < width) {
                val index = (y * width + x) * 4
                val r = rgba[index].toInt() and 0xFF
                val g = rgba[index + 1].toInt() and 0xFF
                val b = rgba[index + 2].toInt() and 0xFF
                val gray = ((r * 30) + (g * 59) + (b * 11)) / 100
                values.add(gray.toByte())
                x += sampleStep
            }
            y += sampleStep
        }
        return values.toByteArray()
    }

    private fun meanAbsoluteDifference(a: ByteArray, b: ByteArray): Float {
        val n = minOf(a.size, b.size)
        if (n == 0) return Float.MAX_VALUE

        var sum = 0f
        for (i in 0 until n) {
            val av = a[i].toInt() and 0xFF
            val bv = b[i].toInt() and 0xFF
            sum += kotlin.math.abs(av - bv).toFloat()
        }
        return sum / n.toFloat()
    }
}
