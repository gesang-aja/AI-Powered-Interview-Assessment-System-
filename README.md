# AI Powered Interview Assessment System
---

## Deskripsi Proyek
AI Powered Interview Assessment System merupakan solusi **end-to-end** yang membantu assessor melakukan validasi pemahaman peserta interview melalui analisis video interview secara otomatis. Sistem aplikasi ini menerima video wawancara sebagai input, melakukan analisis audio dan visual, hingga menghasilkan laporan evaluasi yang objektif dan terstruktur. Sistem ini menyediakan data pendukung sebagai alat bantu penilaian bagi assessor untuk mempercepat dan mempermudah proses evaluasi.
---

## Setup Environment
1. **Prasyarat sistem**
- Python 3.10.x
```bash
python --version
```
- Windows / Linux / macOS

2. **Clone Repository**
- Salin repository proyek ke lingkungan lokal menggunakan Git.
```bash
git clone <repository-url>
cd <nama-folder-proyek>
```

3. **Setup environment**
- Buat virtual environment untuk dependency proyek agar tidak terjadi konflik dengan library lain.
```bash
python -m venv env
```

- Aktivasi environment di dalam environment proyek
```bash
env\Scripts\activate
```

- Install dependency proyek yang dibutuhkan berdasarkan file requirements.txt
```bash
pip install -r requirements.txt
```

- Masukan API KEY ke file .env.example untuk menggunakan LLM Gemini

- Menonaktifkan environment(Opsional)    
```bash
deactivate
```

## Intruksi Demonstrasi Aplikasi
1. **Pastikan virtual environment aktif**
```bash
env\Scripts\activate
```
2. **Masuk direktori proyek**
```bash
cd <nama-folder-proyek>
```
3. **Jalankan streamlit**
```bash
streamlit run app.py
```
4. **Akses aplikasi di browser**
```arduino
http://localhost:8501
```
5. **Menghentikan aplikasi streamlit**
```text
Ctrl + C
```
