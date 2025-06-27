<?php
require_once __DIR__ . '/../config/database.php';

function registerStudent($nim, $name, $photoFilename) {
    global $pdo;
    $stmt = $pdo->prepare("INSERT INTO students (nim, name, photo) VALUES (?, ?, ?)");
    return $stmt->execute([$nim, $name, $photoFilename]);
}

function checkIn($studentId, $notes = null) {
    global $pdo;
    $stmt = $pdo->prepare("INSERT INTO attendance (student_id, check_in, notes) VALUES (?, NOW(), ?)");
    return $stmt->execute([$studentId, $notes]);
}

function checkOut($studentId) {
    global $pdo;
    $stmt = $pdo->prepare("UPDATE attendance SET check_out = NOW() 
                          WHERE student_id = ? AND DATE(check_in) = CURDATE() AND check_out IS NULL");
    return $stmt->execute([$studentId]);
}

function getAttendanceRecords($filter = []) {
    global $pdo;
    $date = $filter['date'] ?? date('Y-m-d');
    // Perbaiki bagian JOIN berikut:
    $stmt = $pdo->prepare("
        SELECT a.*, s.name 
        FROM attendance a
        JOIN students s ON a.nim = s.nim
        WHERE a.date = ?
        ORDER BY a.check_in ASC
    ");
    $stmt->execute([$date]);
    return $stmt->fetchAll(PDO::FETCH_ASSOC);
}

function adminLogin($username, $password) {
    global $pdo;
    $stmt = $pdo->prepare("SELECT * FROM admins WHERE username = ?");
    $stmt->execute([$username]);
    $admin = $stmt->fetch(PDO::FETCH_ASSOC);
    
    if ($admin && password_verify($password, $admin['password'])) {
        return $admin;
    }
    return false;
}
?>
