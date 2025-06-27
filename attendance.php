<?php
require_once __DIR__ . '/includes/functions.php';

header('Content-Type: application/json');

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $identity = $_POST['identity'] ?? '';
    if (!$identity) {
        echo json_encode(['status' => 'error', 'message' => 'No identity provided']);
        exit;
    }

    // Ambil NIM dari nama file, misal: faces/12345678.jpg
    $nim = pathinfo($identity, PATHINFO_FILENAME);

    // Cek apakah NIM terdaftar
    $stmt = $pdo->prepare("SELECT * FROM students WHERE nim = ?");
    $stmt->execute([$nim]);
    $student = $stmt->fetch();

    if (!$student) {
        echo json_encode(['status' => 'error', 'message' => 'Mahasiswa tidak ditemukan']);
        exit;
    }

    // Simpan presensi (check-in/check-out)
    $today = date('Y-m-d');
    $now = date('Y-m-d H:i:s');

    // Cek apakah sudah check-in hari ini
    $stmt = $pdo->prepare("SELECT * FROM attendance WHERE nim = ? AND date = ?");
    $stmt->execute([$nim, $today]);
    $attendance = $stmt->fetch();

    if (!$attendance) {
        // Belum check-in, simpan check-in
        $stmt = $pdo->prepare("INSERT INTO attendance (nim, name, date, check_in) VALUES (?, ?, ?, ?)");
        $stmt->execute([$nim, $student['name'], $today, $now]);
        echo json_encode(['status' => 'success', 'message' => 'Check-in berhasil', 'nim' => $nim, 'name' => $student['name']]);
    } else if (!$attendance['check_out']) {
        // Sudah check-in, simpan check-out
        $stmt = $pdo->prepare("UPDATE attendance SET check_out = ? WHERE id = ?");
        $stmt->execute([$now, $attendance['id']]);
        echo json_encode(['status' => 'success', 'message' => 'Check-out berhasil', 'nim' => $nim, 'name' => $student['name']]);
    } else {
        // Sudah check-in dan check-out
        echo json_encode(['status' => 'info', 'message' => 'Presensi hari ini sudah lengkap', 'nim' => $nim, 'name' => $student['name']]);
    }
}