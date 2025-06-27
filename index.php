<?php
require_once __DIR__ . '/includes/functions.php';

$students = [];
$attendance = [];

if (isset($_GET['search'])) {
    $students = $pdo->query("SELECT * FROM students ORDER BY name")->fetchAll();
    $attendance = getAttendanceRecords(['date' => date('Y-m-d')]);
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SILAB TIF UNISSULA</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="assets/css/style.css" rel="stylesheet">
    <style>
    body { background: #f8f9fa; }
    .video-container { position: relative; width: 100%; }
    #video, #canvas {
        width: 100%;
        height: auto;
        display: block;
    }
    #canvas {
        position: absolute;
        top: 0; left: 0;
        pointer-events: none;
    }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="#">SILAB TIF UNISSULA</a>
            <a href="login.php" class="btn btn-outline-light ms-auto">Admin Login</a>
        </div>
    </nav>

    <div class="container my-5">
        <div class="row">
            <div class="col-md-6">
                <div class="card mb-4">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0">Presensi Masuk/Keluar</h5>
                    </div>
                    <div class="card-body">
                        <div class="video-container mb-3">
                            <video id="video" autoplay muted></video>
                            <canvas id="canvas"></canvas>
                        </div>
                        <div id="detection-status" class="alert alert-info">Camera is loading...</div>
                        <button id="capture-btn" class="btn btn-primary w-100">Detect Face</button>
                        <div id="result" class="mt-3"></div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card mb-4">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0">Registrasi Mahasiswa</h5>
                    </div>
                    <div class="card-body">
                        <a href="register.php" class="btn btn-primary w-100">Register New Student</a>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header bg-primary text-white d-flex justify-content-between">
                        <h5 class="mb-0">Today's Attendance</h5>
                        <form method="get" class="d-inline">
                            <button type="submit" name="search" class="btn btn-sm btn-light">Refresh</button>
                        </form>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>NIM</th>
                                        <th>Name</th>
                                        <th>Check In</th>
                                        <th>Check Out</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <?php foreach ($attendance as $record): ?>
                                    <tr>
                                        <td><?= htmlspecialchars($record['nim']) ?></td>
                                        <td><?= htmlspecialchars($record['name']) ?></td>
                                        <td><?= date('H:i', strtotime($record['check_in'])) ?></td>
                                        <td><?= $record['check_out'] ? date('H:i', strtotime($record['check_out'])) : '-' ?></td>
                                    </tr>
                                    <?php endforeach; ?>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/face-api.js"></script>
<script>
window.onload = function() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const detectionStatus = document.getElementById('detection-status');
    const resultDiv = document.getElementById('result');
    const captureBtn = document.getElementById('capture-btn');

    // Start webcam
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
            detectionStatus.textContent = "Camera ready. Silakan klik Detect Face.";
            detectionStatus.className = "alert alert-success";
            captureBtn.disabled = false; // Aktifkan tombol setelah kamera siap
        })
        .catch(err => {
            detectionStatus.textContent = "Tidak dapat mengakses kamera.";
            detectionStatus.className = "alert alert-danger";
            captureBtn.disabled = true;
        });

    // Load face-api.js models lalu mulai live detection
    Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri('https://cdn.jsdelivr.net/npm/face-api.js/weights')
    ]).then(() => {
        video.addEventListener('loadeddata', () => {
            const displaySize = { width: video.videoWidth, height: video.videoHeight };
            canvas.width = displaySize.width;
            canvas.height = displaySize.height;
            setInterval(async () => {
                const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions());
                const resized = faceapi.resizeResults(detections, displaySize);
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                faceapi.draw.drawDetections(canvas, resized);
            }, 100);
        });
    });

    // Capture & send to DeepFace backend
    captureBtn.addEventListener('click', function() {
        if (video.videoWidth === 0 || video.videoHeight === 0) {
            detectionStatus.textContent = "Kamera belum siap.";
            detectionStatus.className = "alert alert-danger";
            return;
        }
        detectionStatus.textContent = "Mendeteksi wajah...";
        detectionStatus.className = "alert alert-info";
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg');

        fetch('http://localhost:5000/verify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData.split(',')[1] })
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === "success") {
                detectionStatus.textContent = "Wajah dikenali!";
                detectionStatus.className = "alert alert-success";
                resultDiv.innerHTML = "Identitas: <b>" + data.identity + "</b>";
                if (data.framed_image) {
                    resultDiv.innerHTML += `<br><img src="data:image/jpeg;base64,${data.framed_image}" style="width:100%;margin-top:10px;">`;
                }
                fetch('attendance.php', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: 'identity=' + encodeURIComponent(data.identity)
                })
                .then(res => res.json())
                .then(presensi => {
                    if (presensi.status === "success") {
                        resultDiv.innerHTML += "<br><span class='text-success'>" + presensi.message + "</span>";
                    } else if (presensi.status === "info") {
                        resultDiv.innerHTML += "<br><span class='text-info'>" + presensi.message + "</span>";
                    } else {
                        resultDiv.innerHTML += "<br><span class='text-danger'>" + presensi.message + "</span>";
                    }
                });
            } else {
                detectionStatus.textContent = "Wajah tidak dikenali.";
                detectionStatus.className = "alert alert-warning";
                resultDiv.innerHTML = "";
            }
        })
        .catch(() => {
            detectionStatus.textContent = "Gagal mengirim ke server DeepFace.";
            detectionStatus.className = "alert alert-danger";
            resultDiv.innerHTML = "";
        });
    });
};
</script>
</body>
</html>