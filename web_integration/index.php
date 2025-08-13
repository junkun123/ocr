<?php
if (
    $_SERVER['REQUEST_METHOD'] === 'POST' && isset($_FILES['imagen'])
) {
    $tmp = $_FILES['imagen']['tmp_name'];
    $cfile = new CURLFile($tmp, $_FILES['imagen']['type'], $_FILES['imagen']['name']);
    $ch = curl_init('http://127.0.0.1:5000/ocr');
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, ['image' => $cfile]);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    $resp = curl_exec($ch);
    curl_close($ch);
    echo '<pre>' . htmlspecialchars($resp) . '</pre>';
}
?>
<form method="POST" enctype="multipart/form-data">
    <input type="file" name="imagen" accept="image/*" required>
    <button type="submit">Subir y extraer</button>
</form>