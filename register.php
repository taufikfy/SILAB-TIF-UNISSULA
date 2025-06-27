<?php
require_once __DIR__ . '/includes/functions.php';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $nim = $_POST['nim'];
    $name = $_POST['name'];
    $faceData = $_POST['face_data'];

    // Cek apakah NIM sudah terdaftar
    $stmt = $pdo->prepare("SELECT COUNT(*) FROM students WHERE nim = ?");
    $stmt->execute([$nim]);
    if ($stmt->fetchColumn() > 0) {
        $error = "NIM sudah terdaftar. Silakan gunakan NIM lain.";
    } else {
        // Simpan foto ke folder uploads/
        $photoFilename = $nim . '_' . time() . '.jpg';
        $photoPath = __DIR__ . '/faces/' . $photoFilename;

        // Simpan file foto
        $faceData = str_replace('data:image/jpeg;base64,', '', $faceData);
        $faceData = str_replace(' ', '+', $faceData);
        file_put_contents($photoPath, base64_decode($faceData));

        // Simpan ke database
        if (registerStudent($nim, $name, $photoFilename)) {
            header("Location: index.php?registered=1");
            exit;
        } else {
            $error = "Gagal menyimpan ke database.";
        }
    }
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Register Student</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="index.php">Face Attendance</a>
        </div>
    </nav>
    <div class="container my-5">
        <div class="row justify-content-center">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0">Register New Student</h5>
                    </div>
                    <div class="card-body">
                        <form method="post" id="registration-form">
                            <div class="mb-3">
                                <label for="nim" class="form-label">NIM</label>
                                <input type="text" class="form-control" id="nim" name="nim" required>
                            </div>
                            <div class="mb-3">
                                <label for="name" class="form-label">Full Name</label>
                                <input type="text" class="form-control" id="name" name="name" required>
                            </div>
                            <div class="mb-3">
                                <label class="form-label">Face Registration</label>
                                <div class="video-container mb-3">
                                    <video id="video" autoplay muted style="width:100%;"></video>
                                    <canvas id="canvas" class="d-none"></canvas>
                                </div>
                                <div id="detection-status" class="alert alert-info">Camera is loading...</div>
                                <button type="button" id="capture-btn" class="btn btn-primary">Capture Face</button>
                            </div>
                            <input type="hidden" id="face_data" name="face_data">
                            <button type="submit" id="register-btn" class="btn btn-success w-100 mt-3" disabled>Register Student</button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
    // Webcam setup
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const detectionStatus = document.getElementById('detection-status');
    const captureBtn = document.getElementById('capture-btn');
    const faceDataInput = document.getElementById('face_data');
    const registerBtn = document.getElementById('register-btn');

    // Start webcam
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
            detectionStatus.textContent = "Camera ready. Silakan klik Capture Face.";
            detectionStatus.className = "alert alert-success";
        })
        .catch(err => {
            detectionStatus.textContent = "Tidak dapat mengakses kamera.";
            detectionStatus.className = "alert alert-danger";
        });

    // Capture face
    captureBtn.addEventListener('click', function() {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg');
        faceDataInput.value = imageData;
        detectionStatus.textContent = "Foto wajah berhasil diambil.";
        detectionStatus.className = "alert alert-success";
        registerBtn.disabled = false;
    });
    </script>
</body>
</html>