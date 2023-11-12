# Customer-Type-and-Behaviour-Analysis-on-Advertisement

## Latar Belakang
Sebuah perusahaan di Indonesia ingin mengetahui efektifitas sebuah iklan yang mereka tayangkan, hal ini penting bagi perusahaan agar dapat mengetahui seberapa besar ketercapainnya iklan yang dipasarkan sehingga dapat menarik customers untuk melihat iklan.
Dengan mengolah data historical advertisement serta menemukan insight serta pola yang terjadi, maka dapat membantu perusahaan dalam menentukan target marketing, fokus case ini adalah membuat model machine learning classification yang berfungsi menentukan target customers yang tepat

## Objective
Buat model machine learning yang dapat mengklasifikasi konsumen potensial. Setelah didapatkan hasil modeling kita dapat mengetahui fitur importance yang paling berpengaruh terhadap target, selain itu kita dapat menghitung jumlah revenue yang didapatkan sebelum dan sesudah menerapkan machine learning

## Business Metrics
Click Through Rate, Revenue, Segmentasi konsumen,  Faktor penting yang mempengaruhi 

## Dataset
Data yang digunakan dalam analisis tersedia dalam berkas "Clicked Ads Dataset.csv". Yang terdiri dari 1000 baris, 11 kolom

## Exploratory Data Analysis (EDA)
### Statistical analysis

![1  Statistical analysis 1](https://github.com/GedeWiraWasistha/Customer-Type-and-Behaviour-Analysis-on-Advertisement/assets/149849772/1fbae0da-4b20-40e3-8093-db5f9edb5665) <br>
Dataset terdiri dari 1000 baris dan 11 kolom. Beberapa kolom memiliki nilai null, termasuk Daily Time Spent on Site, Area Income, Daily Internet Usage, dan Male. 

![2  Statistical analysis 2](https://github.com/GedeWiraWasistha/Customer-Type-and-Behaviour-Analysis-on-Advertisement/assets/149849772/19fdc5c0-3cdf-49c1-bf17-08763cb223ff) <br>
Seluruh kolom numerik menunjukkan skewness, meskipun tidak secara ekstrem. Analisis lebih lanjut diperlukan untuk memahami implikasi dan tindakan yang dapat diambil.

### Univariate Analysis

![3  skewness](https://github.com/GedeWiraWasistha/Customer-Type-and-Behaviour-Analysis-on-Advertisement/assets/149849772/28c65f9d-9c7d-4a4b-92ab-4fad15428e6f) <br>
Hampir semua kolom numerical menunjukkan skewness, meskipun tidak secara ekstrem. Analisis lebih lanjut diperlukan untuk memahami implikasi dan tindakan yang dapat diambil.

![4  outliers](https://github.com/GedeWiraWasistha/Customer-Type-and-Behaviour-Analysis-on-Advertisement/assets/149849772/cc176e7f-4277-4c2c-bede-cec6d0cdf460) <br>
Teridentifikasi outliers pada kolom Area Income. Diperlukan penanganan untuk menjaga keakuratan analisis data.

### Bivariate Analysis
![5  Bivariate Analysis](https://github.com/GedeWiraWasistha/Customer-Type-and-Behaviour-Analysis-on-Advertisement/assets/149849772/9d9e44be-b092-4e91-98f0-2632fb03ae27) <br>
![6  Bivariate Analysis 2](https://github.com/GedeWiraWasistha/Customer-Type-and-Behaviour-Analysis-on-Advertisement/assets/149849772/28593603-462c-4f79-b730-c806be03136e) <br>

- Konsumen yang mengklik iklan di situs web memiliki usia rata-rata 25–40 tahun. <br>
- Sebaran penggunaan internet harian, potensi pengguna mengklik suatu produk lebih tinggi pada pengguna yang jarang menggunakan internet dibandingkan dengan yang sering menggunakan internet. <br>
- Distribusi waktu harian yang dihabiskan di situs, potensi pengguna mengklik suatu produk lebih tinggi pada pengguna yang menghabiskan lebih sedikit waktu berkunjung. <br> 
- Dari hasil scatterplot antara penggunaan internet sehari-hari dengan waktu yang dihabiskan sehari-hari di situs menunjukkan pola yang terbagi menjadi 2 segmen yaitu pengguna aktif dan pengguna tidak aktif, dimana pengguna aktif cenderung lebih kecil kemungkinannya untuk mengklik iklan di situs web dibandingkan pengguna non-aktif.<br>


### Multivariate Analysis
![7  Multivariate Analysis](https://github.com/GedeWiraWasistha/Customer-Type-and-Behaviour-Analysis-on-Advertisement/assets/149849772/cf7477d4-466d-4978-8355-92fc97bed656) <br>
- Kolom target adalah kolom cliked on ads. <br>
- Kolom fitur yang mempunyai kolerasi yang tinggi dan positif terhadap target adalah kolom Age. sementara kolom Daily Time Spent On Site, Area Income, dan Daily Internet Usage memiliki kolerasi tinggi dan negatif. <br>
- Sementara itu kolerasi tinggi dan positif antar kolom ditunjukan oleh kolom Daily Internet Usage dengan Daily Time Spent On Site, Daily Internet Usage dengan Area Income, Daily Time Spent On Site dengan Area income. <br>
- Sedangkan kolerasi antar fitur yang cukup tinggi dan negatif ditunjukan oleh kolom Daily Time Spent On Site dengan Age, Daily Internet Usage dengan Age <br>

  
## Data Cleaning & Preprocessing
![8  Handling missing value](https://github.com/GedeWiraWasistha/Customer-Type-and-Behaviour-Analysis-on-Advertisement/assets/149849772/48305016-3644-496c-a6da-e1c499415df1) <br>
Dalam proses handling missing value, mean digunakan untuk mengatasi missing value pada kolom Daily Time Spent on Site, Area Income, dan Daily Internet Usage. Sementara itu modus digunakan untuk mengatasi  missing value pada kolom Male, hal ini dikarenakan kolom Male adalah kolom biner.

![9  feature encoding](https://github.com/GedeWiraWasistha/Customer-Type-and-Behaviour-Analysis-on-Advertisement/assets/149849772/86dae6a4-c242-4ee4-8efe-ea71dc68fa86) <br>
Dalam proses feature encoding. Label encoding digunakan pada kolom Click on Ad, dikarenakan kolom ini memiliki nilai biner. Sementara one hot encoding digunakan pada kolom Male dan category, dikarenakan nilai yang terkandung dalam data kolom tersebut adalah non-binari. <br>
Untuk ekstrasi kolom Timestam menjadi beberapa kolom (Year, Month, Week, Day) menggunakan fungsi  to_datetime

![10  Split data](https://github.com/GedeWiraWasistha/Customer-Type-and-Behaviour-Analysis-on-Advertisement/assets/149849772/dab8edf3-fe95-423b-82db-89d07ea6a92e) <br>
Split data training dan data testing menggunakan modul sklearn. Proporsi data training adalah 70% sedangakn untuk data testing sebesar 30%


## Modeling, Evaluasi Model, dan Feature Importance

![11  Modeling, Evaluasi Model, dan Feature Importance 1](https://github.com/GedeWiraWasistha/Customer-Type-and-Behaviour-Analysis-on-Advertisement/assets/149849772/75269ed9-2f54-4d61-b31b-e1bfe76101b8) <br>
![12  Modeling, Evaluasi Model, dan Feature Importance 2](https://github.com/GedeWiraWasistha/Customer-Type-and-Behaviour-Analysis-on-Advertisement/assets/149849772/4d03c05e-efc4-4010-a9c2-5cf7ef53b1f2) <br>
Dalam tahapan kali ini, saya melakukan experiment pada algoritma RandomForestClassifier dan GradientBoostingClassifier yang telah dilakukan hyperparameter tuning. Hasil eksperimen menunjukkan bahwa tidak terdapat perbedaan yang signifikan dalam performa model, baik ketika data telah dinormalisasi maupun tidak, ketika menggunakan algoritma RandomForestClassifier. Namun, perlu diperhatikan bahwa algoritma GradientBoostingClassifier mengalami penurunan kinerja yang cukup kecil saat data telah dinormalisasi. Dikarenakan hasil evaluasi model RandomForestClassifier lebih baik, maka diputuskan jika permodelan menggunakan algoritma RandomForestClassifier 

![13  Confusion Matrix](https://github.com/GedeWiraWasistha/Customer-Type-and-Behaviour-Analysis-on-Advertisement/assets/149849772/bb7a720a-ecd0-4e1a-94f4-d5702c93942d) <br>
Berdasarkan Confusion Matrix, model cenderung memprediksi lebih banyak pengguna yang akan mengklik iklan. Analisis ini didasarkan pada akurasi tinggi dan recall yang optimal. Confusion Matrix yang dihasilkan oleh RandomForestClassifier menunjukkan hasil yang sangat baik dengan sedikit kesalahan prediksi. Ini mengindikasikan bahwa model memberikan hasil evaluasi yang baik dan dapat diandalkan.

![14  Features Importance](https://github.com/GedeWiraWasistha/Customer-Type-and-Behaviour-Analysis-on-Advertisement/assets/149849772/5154a3d9-3166-4d14-855a-a5968fef958e) <br>
Dengan menggunakan model RandomForestClassifier kita dapat melihat feature important dalam membangun model. Berdasarkan metode RandomForestClassifier, kita dapat melihat bahwa penggunaan Daily Internet Uasage dan Daily Time Spent on Site merupakan fitur yang sangat penting dalam menentukan apakah pengguna akan mengklik atau tidak. Fitur penting lainnya adalah Area Income, Age, Day, Month
  
## Rekomendasi Bisnis
- Berdasarkan EDA kami mendapatkan 2 segment konsumen. Segment pertama yaitu pengguna aktif, segment pengguna aktif memiliki ciri-ciri memiliki sering menggunakan internet dan mengunjungi suatu halaman website. Sementara segment kedua adalah pengguna tidak aktif, segment ini memiliki ciri-ciri yang berkebalikan dengan segment pertama <br>
- Jika dirangkum usia menengah, merupakan konsumen tidak aktif, dan rentang income upper middle cendrung lebih tertarik dalam meng-click iklan. Sementara untuk usia muda sampai menengah awal, merupakan konsumen aktif dan memiliki rentang income upper class cenderung tidak meng-clik iklan <br>

## Simulasi Bisnis
### Simulasi Bisnis Tanpa Machine Learning

**Target Variabel: Clicked on Ad**
**Asumsi :**
- Anggaran pengiklanan masing masing konsumen adalah Rp. 1500 <br>
- Jika diasumsikan kita mengimplementasikan simulasi menggunakan dataset awal dengan jumlah konsumen sebesar 1000 orang. Dan masing-masing segment berisikan 500 orang.<br>
- Setiap pengguna yang mengkonversi, maka akan menghasilkan pendapatan sebesar Rp. 5000.<br>
- Tingkat konversion rate sebesar 50 %, karena hanya 500 orang yang mengklik iklan.<br>

**Perhitungan CTR:** <br>
CTR = (n_click : n_tayangan) * 100<br> 
CTR = (500 : 1000) * 100<br> 
CTR = 50%<br> 
**Perhitungan Cost :** <br>
Cost = biaya iklan * n_pengguna<br>
Cost = 1000 * 1500 <br>
Cost = 1500000 <br>
**Perhitungan Revenue :** <br> 
Revenue  = (conversion rate * n_pengguna) * pendapatan<br>
Revenue  = (50% * 1000) * 5000<br> 
Revenue  = 2500000<br> 
**Perhitungan EPS :** <br>
EPS = Revenue / Clicks<br> 
EPS = 2500000 /500<br> 
EPS = 5000<br> 
**Perhitungan Profit :** <br>
Profit = Revenue - Cost<br> 
Profit = 2500000 - 1500000<br> 
Profit = 1000000<br> 

### Simulasi Bisnis Dengan Machine Learning

**Target Variabel: Clicked on Ad**
**Asumsi :**
- Anggaran pengiklanan masing masing konsumen adalah Rp. 1500. <br>
- Berdasarkan kinerja model, model yang telah disepakati mendapatkan akurasi hasil pengujian sebesar 94.28%, sehingga jika diterapkan pada dataset awal, akan mendapatkan 942 pengguna yang melakukan konversi.<br>
- Setiap pengguna yang mengkonversi, maka akan menghasilkan pendapatan sebesar Rp. 5000<br>

**Perhitungan CTR:** <br>
CTR = (n_click : n_tayangan) * 100 <br>
CTR = (942 : 1000) * 100<br>
CTR = 94,2%<br>
**Perhitungan Cost :** <br> 
Cost = biaya iklan * n_pengguna <br>
Cost = 1000 * 1500 <br>
Cost = 1500000 <br>
**Perhitungan Revenue :** <br>
Revenue  = (conversion rate * n_pengguna) * pendapatan <br>
Revenue  = (94,28% * 1000) *  5000 <br>
Revenue  = 4710000 <br>
**Perhitungan EPS :** <br>
EPS = Revenue / Clicks <br>
EPS = 4710000 /942 <br>
EPS = 5000 <br>
**Perhitungan Profit :** <br>
Profit = Revenue - Cost <br>
Profit = 4710000- 1500000 <br>
Profit = 3210000 <br>

## Kesimpulan
Berdasarkan simulasi di atas, jika kita tidak menggunakan model pembelajaran mesin, maka kita akan mendapatkan pendapatan 1 juta dan dengan penggunaan ML pendapatan meningkat secara signifikan lebih dari tiga kali lipat. Kesimpulannya, Machine Learning dapat bekerja dengan baik untuk meningkatkan pendapatan.
