package com.transbloom.cameraapp

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.transbloom.cameraapp.inference.FlowerOnnxClassifier
import com.transbloom.cameraapp.inference.Prediction
import com.transbloom.cameraapp.inference.StabilityState
import com.transbloom.cameraapp.inference.StillFrameAnalyzer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                CameraClassifierApp()
            }
        }
    }
}

@Composable
private fun CameraClassifierApp() {
    val context = LocalContext.current
    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED,
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
    ) { granted ->
        hasPermission = granted
    }

    LaunchedEffect(Unit) {
        if (!hasPermission) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    if (!hasPermission) {
        PermissionScreen(
            onRequestPermission = { permissionLauncher.launch(Manifest.permission.CAMERA) },
        )
    } else {
        CameraClassifierScreen()
    }
}

@Composable
private fun PermissionScreen(onRequestPermission: () -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF10171F)),
        contentAlignment = Alignment.Center,
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = "Camera permission is required to run live flower detection.",
                color = Color.White,
            )
            Button(
                modifier = Modifier.padding(top = 12.dp),
                onClick = onRequestPermission,
            ) {
                Text("Grant Camera Permission")
            }
        }
    }
}

@Composable
private fun CameraClassifierScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val mainHandler = remember { Handler(Looper.getMainLooper()) }

    var statusText by remember { mutableStateOf("Starting camera...") }
    var resultLabel by remember { mutableStateOf("Waiting for still frame...") }
    var confidence by remember { mutableFloatStateOf(0f) }

    val previewView = remember {
        PreviewView(context).apply {
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }
    val cameraExecutor: ExecutorService = remember { Executors.newSingleThreadExecutor() }

    val classifierHolder = remember {
        runCatching { FlowerOnnxClassifier(context) }
    }
    val classifier = classifierHolder.getOrNull()
    val classifierInitError = classifierHolder.exceptionOrNull()

    DisposableEffect(Unit) {
        onDispose {
            classifier?.close()
            cameraExecutor.shutdown()
        }
    }

    if (classifierInitError != null) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black),
            contentAlignment = Alignment.Center,
        ) {
            Text(
                text = "Failed to load ONNX model from assets.\n${classifierInitError.message}",
                color = Color.White,
                modifier = Modifier.padding(24.dp),
            )
        }
        return
    }

    DisposableEffect(previewView, lifecycleOwner, classifier) {
        if (classifier == null) {
            onDispose {}
        } else {
            val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
            val mainExecutor = ContextCompat.getMainExecutor(context)

            cameraProviderFuture.addListener(
                {
                    val cameraProvider = cameraProviderFuture.get()
                    val preview = Preview.Builder()
                        .build()
                        .also { it.surfaceProvider = previewView.surfaceProvider }

                    val analyzer = StillFrameAnalyzer(
                        classifier = classifier,
                        onStabilityUpdate = { state: StabilityState ->
                            mainHandler.post {
                                statusText = formatStabilityText(state)
                            }
                        },
                        onPrediction = { prediction: Prediction ->
                            mainHandler.post {
                                resultLabel = prediction.label
                                confidence = prediction.confidence
                            }
                        },
                    )

                    val imageAnalysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build()

                    imageAnalysis.setAnalyzer(cameraExecutor, analyzer)

                    try {
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            CameraSelector.DEFAULT_BACK_CAMERA,
                            preview,
                            imageAnalysis,
                        )
                        statusText = "Camera running. Hold still to classify."
                    } catch (exc: Exception) {
                        statusText = "Camera bind failed: ${exc.message}"
                    }
                },
                mainExecutor,
            )

            onDispose {
                runCatching {
                    cameraProviderFuture.get().unbindAll()
                }
            }
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { previewView },
        )

        Column(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .fillMaxWidth()
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            OverlayCard(title = "Status", value = statusText)
            OverlayCard(
                title = "Prediction",
                value = "$resultLabel (${(confidence * 100f).toInt()}%)",
            )
        }
    }
}

@Composable
private fun OverlayCard(title: String, value: String) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(text = title)
            Text(text = value)
        }
    }
}

private fun formatStabilityText(state: StabilityState): String {
    if (state.motionScore == Float.MAX_VALUE) {
        return "Sampling frame motion..."
    }
    val stableTag = if (state.isStable) "stable" else "moving"
    return "Motion=${"%.2f".format(state.motionScore)} | stillFrames=${state.stableFrames} ($stableTag)"
}
