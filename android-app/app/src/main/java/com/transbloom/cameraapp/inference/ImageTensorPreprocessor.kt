package com.transbloom.cameraapp.inference

object ImageTensorPreprocessor {
    fun rgbaToNchwFloat32(
        rgba: ByteArray,
        srcWidth: Int,
        srcHeight: Int,
        dstSize: Int = 32,
    ): FloatArray {
        val channelSize = dstSize * dstSize
        val output = FloatArray(3 * channelSize)

        for (y in 0 until dstSize) {
            val srcY = (y * srcHeight) / dstSize
            for (x in 0 until dstSize) {
                val srcX = (x * srcWidth) / dstSize
                val srcIndex = (srcY * srcWidth + srcX) * 4

                val r = (rgba[srcIndex].toInt() and 0xFF) / 255f
                val g = (rgba[srcIndex + 1].toInt() and 0xFF) / 255f
                val b = (rgba[srcIndex + 2].toInt() and 0xFF) / 255f

                val pixelIndex = y * dstSize + x
                output[pixelIndex] = (r - 0.5f) / 0.5f
                output[channelSize + pixelIndex] = (g - 0.5f) / 0.5f
                output[(2 * channelSize) + pixelIndex] = (b - 0.5f) / 0.5f
            }
        }
        return output
    }
}
