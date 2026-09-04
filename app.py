import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROFILE_FOLDER'] = 'static/profiles'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROFILE_FOLDER'], exist_ok=True)

class_names = [
    'Strawberry___Leaf_scorch', 
    'Strawberry___healthy', 
    'gray_mold', 
    'leaf_spot', 
    'powdery_mildew_leaf'
]

DISEASE_INFO = {
    'Strawberry___Leaf_scorch': {
        'title': 'Strawberry Leaf Scorch', 
        'desc': 'Penyakit infeksi jamur berbahaya yang menyerang organ vegetatif tanaman seperti daun, tangkai daun, serta kelopak buah stroberi. Patogen ini mengganggu proses metabolisme dan fotosintesis tanaman secara akumulatif, yang jika dibiarkan tanpa penanganan dapat memicu penurunan tingkat produktivitas dan kualitas hasil panen secara drastis.', 
        'symptom': 'Pada tahap awal, muncul bercak-bercak kecil berwarna ungu tua atau kemerahan pada permukaan atas daun. Bercak ini kemudian meluas, bergabung satu sama lain, berubah warna menjadi kecokelatan dengan bagian tengah yang kering seperti terbakar api. Serangan parah membuat daun mengeriting dan gugur.', 
        'control': '1. Sanitasi kebun dengan membersihkan dan membakar sisa tanaman yang terinfeksi.<br>2. Hindari sistem penyiraman dari atas (overhead sprinkler) yang memercikkan spora.<br>3. Gunakan fungisida berbahan aktif tembaga atau Captan secara preventif.<br>4. Terapkan rotasi tanaman dan jaga jarak antar tanam agar sirkulasi udara lancar.',
        'img': 'Leaf scorch.jpg',
        'detail_img': 'Leaf scorch1.jpg'
    },
    'Strawberry___healthy': {
        'title': 'Strawberry Healthy', 
        'desc': 'Kondisi fisiologis optimal tanaman stroberi yang tumbuh dengan ketersediaan unsur hara makro dan mikro tercukupi, memiliki sistem jaringan sel yang kuat, serta terbebas sepenuhnya dari segala bentuk kolonisasi patogen, jamur, bakteri, maupun gangguan hama perusak.', 
        'symptom': 'Daun berwarna hijau cerah merata, segar, tekstur kenyal dan kokoh, tidak ditemukan adanya bintik nekrotik, lubang, deformasi, ataupun lapisan serbuk asing pada permukaan atas maupun bawah daun.', 
        'control': '1. Pertahankan pemupukan berimbang makro dan mikro.<br>2. Lakukan penyiraman secara teratur di area perakaran.<br>3. Lakukan pemantauan (monitoring) rutin setiap minggu.<br>4. Jaga kebersihan lahan dari gulma dan gulma pengganggu.',
        'img': 'Strawberry Healthy.jpg',
        'detail_img': 'Strawberry healthy1.jpg'
    },
    'alternaria_leaf_blight': {
        'title': 'Alternaria Leaf Blight', 
        'desc': 'Penyakit hawar daun yang disebabkan oleh jamur patogen Alternaria spp., menyerang jaringan sel daun dan melemahkan kapasitas fotosintesis tanaman secara signifikan.', 
        'symptom': 'Muncul bintik kecil berwarna cokelat tua yang perlahan membesar membentuk bercak oval konsentris berukuran besar. Di sekitar bercak seringkali terlihat halo atau lingkaran berwarna kekuningan. Daun yang terserang berat akan mengering, mati, dan gugur sebelum waktunya.', 
        'control': '1. Hindari penyiraman tanaman yang terlalu sore untuk mencegah kelembapan daun semalaman.<br>2. Lakukan pemangkasan pada daun tua yang rimbun guna meningkatkan aerasi.<br>3. Aplikasikan fungisida spektrum luas secara berkala saat musim hujan.<br>4. Pastikan pemupukan Kalium diberikan secara cukup untuk meningkatkan ketahanan dinding sel daun.',
        'img': 'alternaria leaf blight.jpg',
        'detail_img': 'alternaria leaf blight1.jpg'
    },
    'angular_leafspot': {
        'title': 'Angular Leafspot', 
        'desc': 'Penyakit bakteri Xanthomonas fragariae yang sangat merusak, ditandai dengan lesi basah yang dibatasi secara ketat oleh pembuluh tulang daun.', 
        'symptom': 'Pada pagi hari atau kondisi lembap, daun menunjukkan bercak kecil berair yang transparan (water-soaked) berbentuk sudut-sudut kecil mengikuti pola tulang daun. Bila cuaca kering, bercak berubah menjadi kecokelatan mengkilap seperti kertas minyak. Infeksi dapat menyebar ke kelopak bunga.', 
        'control': '1. Gunakan benih atau bibit bersertifikat yang dijamin bebas dari infeksi bakteri.<br>2. Hindari aktivitas perawatan kebun saat kondisi daun masih basah oleh embun atau hujan.<br>3. Semprotkan bakterisida berbahan dasar tembaga (copper bactericide) pada fase awal.<br>4. Karantina area kebun yang terinfeksi agar tidak menyebar ke tanaman sehat.',
        'img': 'angular_leafspot_1.jpg',
        'detail_img': 'angular_leafspot_2.jpg'
    },
    'anthocyanosis': {
        'title': 'Anthocyanosis', 
        'desc': 'Gangguan fisiologis tanaman akibat akumulasi pigmen antosianin yang berlebihan pada jaringan daun dan tangkai sebagai respons terhadap cekaman lingkungan atau ketidakseimbangan nutrisi.', 
        'symptom': 'Seluruh helai daun atau bagian tepinya berubah warna secara tidak normal menjadi kemerahan, keunguan, hingga kebiruan gelap. Hal ini sering disalahartikan sebagai penyakit infeksi padahal merupakan respons stres lingkungan atau hara.', 
        'control': '1. Periksa ketersediaan unsur hara Fosfor (P) dan Nitrogen (N) di dalam tanah.<br>2. Sesuaikan pH tanah agar berada pada kisaran optimal penyerapan mineral (6.0 - 6.5).<br>3. Berikan naungan atau perlindungan jika tanaman mengalami cekaman suhu dingin ekstrem.<br>4. Pastikan drainase media tanam berfungsi dengan baik.',
        'img': 'anthocyanosis_29.jpg',
        'detail_img': 'anthocyanosis_30.jpg'
    },
    'anthracnose': {
        'title': 'Anthracnose', 
        'desc': 'Penyakit tular tanah dan udara yang sangat mematikan pada budidaya stroberi, disebabkan oleh infeksi jamur Colletotrichum spp. Karakteristik utamanya adalah kemampuan penyebaran spora yang sangat cepat melalui percikan air hujan.', 
        'symptom': 'Menimbulkan lesi atau luka cekung berwarna cokelat gelap hingga hitam pada tangkai daun, stolon, dan buah stroberi. Pada buah, lesi berkembang menjadi busuk keras berair yang ditutupi massa spora berwarna merah muda salmon dalam kondisi lembap.', 
        'control': '1. Gunakan bibit bebas penyakit yang bersertifikat.<br>2. Aplikasikan fungisida sistemik dan kontak secara berkala pada fase pembungaan.<br>3. Hindari pemupukan Nitrogen berlebih yang memicu pertumbuhan jaringan lunak.<br>4. Segera petik dan musnahkan buah atau bagian tanaman yang menunjukkan gejala.',
        'img': 'anthracnose_1.jpg',
        'detail_img': 'anthracnose_2.jpg'
    },
    'anthracnose_fruit_rot': {
        'title': 'Anthracnose Fruit Rot', 
        'desc': 'Fase spesifik infeksi antraknosa yang menyerang langsung bagian buah stroberi baik pada fase pematangan maupun pascapanen, menurunkan nilai jual hasil pertanian secara drastis.', 
        'symptom': 'Muncul bintik kecil berwarna cokelat muda yang dengan cepat membesar menjadi lingkaran cekung, kering, dan berwarna gelap pekat. Pada kelembapan tinggi, permukaan cekungan tersebut dipenuhi lendir spora berwarna jingga/merah muda.', 
        'control': '1. Cegah kontak langsung buah dengan permukaan tanah dengan cara memasang mulsa plastik.<br>2. Lakukan pemanenan tepat waktu dan hindari pembiakan buah yang terlalu matang di pohon.<br>3. Semprotkan fungisida protektif pada fase pembentukan buah.<br>4. Simpan hasil panen pada suhu dingin yang terkontrol.',
        'img': 'anthracnose_fruit_rot_10.jpg',
        'detail_img': 'anthracnose_fruit_rot_11.jpg'
    },
    'ants': {
        'title': 'Ants (Semut)', 
        'desc': 'Koloni semut yang membuat sarang di sekitar perakaran tanaman dan kerap membawa serta memelihara hama kutu daun (aphids) untuk mendapatkan cairan madu.', 
        'symptom': 'Tanaman terlihat kurang stabil karena sarang semut melonggarkan media tanam di sekitar perakaran. Selain itu, kehadiran semut dalam jumlah banyak di tangkai daun biasanya menjadi indikator adanya koloni kutu daun.', 
        'control': '1. Taburkan umpan atau pengendali organik ramah lingkungan di jalur keluar masuk semut.<br>2. Siram media tanam dengan larutan air sabun organik jika sarang mengganggu perakaran.<br>3. Basmi sumber hama utama seperti kutu daun agar semut dengan sendirinya pergi.<br>4. Jaga kebersihan sekitar bedengan tanaman.',
        'img': 'ants_21.jpg',
        'detail_img': 'ants_22.jpg'
    },
    'aphid': {
        'title': 'Aphid (Kutu Daun)', 
        'desc': 'Hama serangga bertubuh lunak yang bergerombol di pucuk daun muda dan menghisap cairan nutrisi penting dari pembuluh tanaman.', 
        'symptom': 'Daun muda dan pucuk mengalami keriting, mengkerut, serta pertumbuhan terhambat. Hama ini juga mengeluarkan cairan manis (honeydew) yang memicu timbulnya jamur embun jelaga hitam di permukaan daun.', 
        'control': '1. Semprot menggunakan insektisida nabati ekstrak bawang putih atau minyak neem.<br>2. Gunakan sabun insektisida (insecticidal soap) langsung pada koloni kutu.<br>3. Manfaatkan musuh alami seperti kumbang koksi (ladybug).<br>4. Pasang perangkap kuning lengket di sekitar area tanaman.',
        'img': 'aphid_1.jpg',
        'detail_img': 'aphid_2.jpg'
    },
    'aphid_effects': {
        'title': 'Aphid Effects (Dampak Kutu Daun)', 
        'desc': 'Kondisi kerusakan lanjutan dan sekunder akibat serangan hama kutu daun yang tidak segera ditangani secara preventif.', 
        'symptom': 'Daun mengalami deformasi parah (melengkung dan cacat), tertutup lapisan hitam embun jelaga, serta tanaman menjadi kerdil karena kekurangan nutrisi yang dihisap secara terus-menerus oleh populasi kutu.', 
        'control': '1. Lakukan pemangkasan total pada daun atau tunas yang keriting parah dan rusak.<br>2. Bersihkan lapisan jelaga hitam menggunakan usapan air bersih.<br>3. Basmi tuntas sisa populasi kutu dengan insektisida sistemik.<br>4. Berikan pemupukan susulan pemulihan tanaman.',
        'img': 'aphid_effects_3.jpg',
        'detail_img': 'aphid_effects_4.jpg'
    },
    'ascochyta_blight': {
        'title': 'Ascochyta Blight', 
        'desc': 'Penyakit jamur hawar ascochyta yang menyerang jaringan daun dan batang stroberi pada kondisi cuaca basah dan lembap.', 
        'symptom': 'Timbul bercak berbentuk tidak beraturan berwarna cokelat muda hingga kelabu terang pada helaian daun. Di dalam area bercak tersebut, seringkali tampak bintik-bintik hitam kecil yang merupakan tubuh buah jamur (piknidia).', 
        'control': '1. Lakukan pergiliran tanaman dengan jenis hortikultura bukan sefamili.<br>2. Buang dan musnahkan daun yang menunjukkan gejala bercak.<br>3. Aplikasikan fungisida berbahan aktif mancozeb atau chlorothalonil.<br>4. Kurangi kelembapan lingkungan sekitar tajuk tanaman.',
        'img': 'ascochyta_blight_3.jpg',
        'detail_img': 'ascochyta_blight_4.jpg'
    },
    'bacterial_spot': {
        'title': 'Bacterial Spot', 
        'desc': 'Infeksi bakteri patogen yang menimbulkan bercak-bercak kecil berair dan berpenetrasi ke dalam jaringan internal daun stroberi.', 
        'symptom': 'Bercak awal tampak kecil, basah, dan dibatasi oleh urat daun. Lama-kelamaan bercak berubah warna menjadi cokelat tua atau kehitaman. Dalam kondisi parah, bercak dapat bersatu menyebabkan kematian jaringan daun secara luas.', 
        'control': '1. Jaga kebersihan alat pertanian agar tidak menularkan bakteri antar tanaman.<br>2. Aplikasikan bakterisida pertanian berbahan tembaga sesuai takaran anjuran.<br>3. Hindari penanaman terlalu rapat yang menghalangi sirkulasi angin.<br>4. Musnahkan tanaman yang terserang berat.',
        'img': 'bacterial_spot_30.jpg',
        'detail_img': 'bacterial_spot_31.jpg'
    },
    'black_chaff': {
        'title': 'Black Chaff', 
        'desc': 'Penyakit yang memicu timbulnya garis-garis atau bercak kehitaman pada jaringan organ tanaman akibat infeksi patogen spesifik.', 
        'symptom': 'Terdapat garis-garis lesi berwarna gelap atau hitam kecokelatan pada tangkai daun atau bagian penumpu daun, mengganggu penyerapan air dan nutrisi dari batang utama.', 
        'control': '1. Sanitasi lahan secara menyeluruh setelah masa panen selesai.<br>2. Gunakan varietas tanaman yang memiliki ketahanan terhadap patogen garis hitam.<br>3. Jaga keseimbangan nutrisi tanah.',
        'img': 'black_chaff_29.jpg',
        'detail_img': 'black_chaff_30.jpg'
    },
    'black_rot': {
        'title': 'Black Rot', 
        'desc': 'Penyakit busuk hitam destruktif yang menyerang sistem jaringan pengangkutan air serta organ bawah tanaman.', 
        'symptom': 'Jaringan akar dan pangkal batang berubah warna menjadi hitam legam, melunak, dan berbau busuk. Daun tanaman mendadak layu, menguning, dan mati karena pasokan air terhambat total.', 
        'control': '1. Lakukan sterilisasi media tanam sebelum penanaman bibit baru.<br>2. Hindari genangan air di sekitar pangkal batang (sistem drainase harus baik).<br>3. Gunakan fungisida atau agen hayati pengandali akar.<br>4. Cabut dan bakar tanaman yang terkena busuk hitam total.',
        'img': 'black_rot_25.jpg',
        'detail_img': 'black_rot_26.jpg'
    },
    'black_spots': {
        'title': 'Black Spots', 
        'desc': 'Kelompok gejala bercak hitam soliter yang muncul akibat berbagai jenis spora jamur oportunis di lingkungan lembap.', 
        'symptom': 'Bercak-bercak hitam soliter dengan batas tegas tersebar di berbagai sisi permukaan daun, menurunkan luas area hijau untuk proses fotosintesis.', 
        'control': '1. Hindari pelukaan mekanis pada daun saat melakukan penyiangan.<br>2. Lakukan penyemprotan fungisida pencegah jamur secara rutin.<br>3. Jaga kebersihan kebun dari gulma.',
        'img': 'black_spots_38.jpg',
        'detail_img': 'black_spots_39.jpg'
    },
    'blossom_blight': {
        'title': 'Blossom Blight', 
        'desc': 'Penyakit hawar bunga yang menyerang organ reproduksi tanaman stroberi sehingga gagal melakukan penyerbukan dan membentuk buah.', 
        'symptom': 'Kelopak dan mahkota bunga berubah warna menjadi cokelat gelap, layu, dan membusuk sebelum sempat mekar atau diserbuki. Sering kali ditemukan lapisan jamur abu-abu tipis di kelopak bunga yang terserang.', 
        'control': '1. Semprotkan fungisida protektif khusus pada fase sebelum dan selama berbunga.<br>2. Pangkas bunga yang sudah menunjukkan gejala busuk.<br>3. Kurangi kelembapan area budidaya.',
        'img': 'blossom_blight_85.jpg',
        'detail_img': 'blossom_blight_86.jpg'
    },
    'blossom_end_rot': {
        'title': 'Blossom End Rot', 
        'desc': 'Gangguan fisiologis berupa busuk ujung buah yang dipicu oleh defisiensi kalsium serta fluktuasi pasokan air yang tidak teratur.', 
        'symptom': 'Bagian ujung atau pantat buah stroberi yang sedang berkembang cekung, berubah warna menjadi cokelat tua hingga hitam legam, dan mengering seperti kulit.', 
        'control': '1. Berikan pemupukan kalsium secara rutin melalui tanah atau semprotan daun.<br>2. Jaga konsistensi kelembapan tanah (jangan biarkan terlalu kering lalu basah mendadak).<br>3. Perbaiki sistem irigasi tetes di lahan.',
        'img': 'blossom_end_rot_7.jpg',
        'detail_img': 'blossom_end_rot_8.jpg'
    },
    'botrytis_cinerea': {
        'title': 'Botrytis Cinerea (Gray Mold)', 
        'desc': 'Jamur busuk abu-abu polifag yang sangat agresif menyerang buah matang, bunga, serta daun tua stroberi pada cuaca basah.', 
        'symptom': 'Jaringan tanaman melunak, berubah kecokelatan, lalu ditumbuhi lapisan tebal spora jamur berwarna abu-abu bertepung yang mudah terbang bila tertiup angin.', 
        'control': '1. Kurangi kelembapan dengan jarak tanam yang tidak terlalu padat.<br>2. Segera panen buah yang sudah matang dan singkirkan buah busuk.<br>3. Aplikasikan fungisida spesifik botrytis pada saat cuaca mendung/hujan.',
        'img': 'botrytis_cinerea_15.jpg',
        'detail_img': 'botrytis_cinerea_16.jpg'
    },
    'burn': {
        'title': 'Burn (Terbakar / Scorch)', 
        'desc': 'Kerusakan fisiologis pada jaringan daun akibat sengatan terik matahari berlebih atau kelebihan konsentrasi pupuk kimia.', 
        'symptom': 'Bagian ujung dan pinggiran helai daun tampak kering, berwarna cokelat tua, dan rapuh seperti kertas terbakar. Tidak ada tanda-tanda spora jamur atau aktivitas hama di area yang rusak.', 
        'control': '1. Berikan peneduh (net/paranet) jika intensitas matahari terlalu terik.<br>2. Sesuaikan takaran dosis pemupukan agar tidak terlalu pekat.<br>3. Lakukan penyiraman air bersih yang cukup untuk mencuci residu pupuk di tanah.',
        'img': 'burn_7.jpg',
        'detail_img': 'burn_8.jpg'
    },
    'calciumdeficiency': {
        'title': 'Calcium Deficiency', 
        'desc': 'Defisiensi unsur hara kalsium yang menghambat pembentukan dinding sel baru pada titik tumbuh dan jaringan meristematik.', 
        'symptom': 'Ujung dan pinggir daun muda mengalami klorosis, melengkung ke bawah, mengering, serta jaringan daun tampak cacat atau robek di bagian tepinya.', 
        'control': '1. Berikan suplai pupuk kalsium nitrat atau kalsium klorida secukupnya.<br>2. Periksa tingkat keasaman (pH) tanah agar unsur kalsium dapat terserap optimal oleh akar.',
        'img': 'calciumdeficiency_9.jpg',
        'detail_img': 'calciumdeficiency_10.jpg'
    },
    'canker': {
        'title': 'Canker (Kanker Batang)', 
        'desc': 'Luka infeksi kronis pada jaringan kayu atau batang utama tanaman yang menghambat translokasi nutrisi.', 
        'symptom': 'Timbul cekungan luka mati (canker) pada batang, terkadang disertai retakan kulit kayu dan keluarnya cairan getah. Bagian tanaman di atas luka akan layu perlahan.', 
        'control': '1. Kerok bagian batang yang terkena kanker lalu oleskan fungisida/bakterisida.<br>2. Hindari pelukaan batang saat merawat tanaman.<br>3. Bakar sisa potongan batang sakit.',
        'img': 'canker_3.jpg',
        'detail_img': 'canker_4.jpg'
    },
    'caterpillars': {
        'title': 'Caterpillars (Ulat Pemakan Daun)', 
        'desc': 'Larva serangga hama dari kelompok lepidoptera yang aktif memakan helaian daun stroberi secara masif.', 
        'symptom': 'Helaian daun berlubang-lubang besar, bergerigi tidak beraturan, atau bahkan hanya tersisa tulang daunnya saja akibat lahapnya gigitan ulat.', 
        'control': '1. Lakukan pengumpulan dan pemusnahan ulat secara manual pada pagi/sore hari.<br>2. Semprotkan insektisida biologi berbasis Bacillus thuringiensis (Bt).<br>3. Pangkas gulma sekitar lahan yang menjadi tempat persembunyian ngengat.',
        'img': 'caterpillars_33.jpg',
        'detail_img': 'caterpillars_34.jpg'
    },
    'cherry_leaf_spot': {
        'title': 'Cherry Leaf Spot', 
        'desc': 'Infeksi bintik daun sekunder yang dapat menyerang berbagai tanaman hortikultura termasuk komoditas stroberi.', 
        'symptom': 'Bintik-bintik kecil berwarna kemerahan atau kecokelatan yang tersebar merata di seluruh permukaan helai daun.', 
        'control': '1. Sanitasi daun gugur di sekitar tanaman.<br>2. Aplikasikan fungisida pelindung.',
        'img': 'cherry_leaf_spot_6.jpg',
        'detail_img': 'cherry_leaf_spot_7.jpg'
    },
    'coccomyces_of_pome_fruits': {
        'title': 'Coccomyces', 
        'desc': 'Infeksi jamur patogen yang menghasilkan bintik bercak daun berpola khas pada tanaman buah.', 
        'symptom': 'Muncul bintik kecil bertepung di sisi bawah daun yang disertai perubahan warna jaringan atas daun.', 
        'control': '1. Semprot dengan fungisida sistemik.<br>2. Jaga kebersihan lahan.',
        'img': 'coccomyces_of_pome_fruits_9.jpg',
        'detail_img': 'coccomyces_of_pome_fruits_10.jpg'
    },
    'colorado_beetle': {
        'title': 'Colorado Beetle', 
        'desc': 'Hama kumbang pemakan daun dengan pola garis khas yang merusak tajuk tanaman.', 
        'symptom': 'Daun rusak parah karena digigit oleh kumbang dewasa maupun larva yang bernafsu makan tinggi.', 
        'control': '1. Gunakan insektisida selektif.<br>2. Pasang perangkap fisik.',
        'img': 'colorado_beetle_11.jpg',
        'detail_img': 'colorado_beetle_12.jpg'
    },
    'downy_mildew': {
        'title': 'Downy Mildew (Embun Tepung Palsu)', 
        'desc': 'Penyakit jamur air oomycete yang memicu bercak kuning di permukaan atas dan jamur putih keabuan di bawah daun.', 
        'symptom': 'Permukaan atas daun menunjukkan bercak kuning dibatasi tulang daun, sedangkan bagian bawahnya berbulu halus keputihan.', 
        'control': '1. Gunakan fungisida sistemik spesifik oomycete.<br>2. Perbaiki drainase lahan.',
        'img': 'downy_mildew_6.jpg',
        'detail_img': 'downy_mildew_7.jpg'
    },
    'cyclamen_mite': {
        'title': 'Cyclamen Mite', 
        'desc': 'Tungau mikroskopis yang sangat menyukai pucuk daun muda dan tunas baru tanaman stroberi.', 
        'symptom': 'Daun baru yang tumbuh menjadi kaku, menebal, melengkung ke bawah, dan berwarna keperakan.', 
        'control': '1. Aplikasikan akarisida khusus tungau.<br>2. Hindari bibit terkontaminasi.',
        'img': 'cyclamen_mite_21.jpg',
        'detail_img': 'cyclamen_mite_22.jpg'
    },
    'dry_rot': {
        'title': 'Dry Rot (Busuk Kering)', 
        'desc': 'Penyakit pembusukan yang menyebabkan jaringan organ tanaman mengering total tanpa mengeluarkan lendir air.', 
        'symptom': 'Bagian buah atau umbi/akar mengering, berkerut, dan keras seperti kayu.', 
        'control': '1. Simpan hasil panen di tempat kering.<br>2. Buang bagian busuk.',
        'img': 'dry_rot_6.jpg',
        'detail_img': 'dry_rot_19.jpg'
    },
    'edema': {
        'title': 'Edema', 
        'desc': 'Kelainan fisiologis akibat penumpukan tekanan air internal daun karena kelembapan tanah dan udara yang terlalu tinggi.', 
        'symptom': 'Timbul bintik-bintik melepuh seperti kutil kecil di permukaan bawah daun yang lama-kelamaan mengering dan pecah.', 
        'control': '1. Kurangi volume dan frekuensi penyiraman air.<br>2. Perbaiki ventilasi udara.',
        'img': 'edema_26.jpg',
        'detail_img': 'edema_27.jpg'
    },
    'esca': {
        'title': 'Esca', 
        'desc': 'Kompleks penyakit jamur kayu yang merusak jaringan pembuluh internal tanaman.', 
        'symptom': 'Daun memperlihatkan pola belang kuning atau garis nekrotik seperti harimau.', 
        'control': '1. Potong dan bakar bagian yang sakit parah.',
        'img': 'esca_2.jpg',
        'detail_img': 'esca_3.jpg'
    },
    'eyespot': {
        'title': 'Eyespot', 
        'desc': 'Penyakit bercak daun yang memiliki bentuk menyerupai mata.', 
        'symptom': 'Bercak bulat dengan pusat terang dan cincin tepi gelap.', 
        'control': '1. Semprot fungisida rutin.',
        'img': 'eyespot_1.jpg',
        'detail_img': 'eyespot_2.jpg'
    },
    'frost_cracks': {
        'title': 'Frost Cracks', 
        'desc': 'Kerusakan fisik jaringan tanaman akibat paparan suhu beku ekstrem.', 
        'symptom': 'Batang atau daun mengalami retakan memanjang.', 
        'control': '1. Berikan perlindungan sungkup plastik saat suhu dingin.',
        'img': 'frost_cracks_21.jpg',
        'detail_img': 'frost_cracks_22.jpg'
    },
    'galls': {
        'title': 'Galls (Bengkak / Kutil)', 
        'desc': 'Benjolan abnormal akibat rangsangan patogen bakteri atau serangga pembentuk galls.', 
        'symptom': 'Terdapat bengkak atau tumor kecil pada batang dan tangkai daun.', 
        'control': '1. Potong jaringan bengkak dan oleskan disinfektan.',
        'img': 'galls_1.jpg',
        'detail_img': 'galls_2.jpg'
    },
    'gray_mold': {
        'title': 'Gray Mold', 
        'desc': 'Penyakit busuk abu-abu pada daun dan buah stroberi akibat kelembapan tinggi.', 
        'symptom': 'Jaringan melunak dan ditumbuhi spora abu-abu bertepung.', 
        'control': '1. Kurangi kelembapan dan pangkas daun lebat.',
        'img': 'gray_mold_42.jpg',
        'detail_img': 'gray_mold_43.jpg'
    },
    'grey_mold': {
        'title': 'Grey Mold', 
        'desc': 'Variasi penamaan jamur botrytis penyebab busuk abu-abu.', 
        'symptom': 'Pembusukan buah disertai jamur lembut abu-abu.', 
        'control': '1. Gunakan fungisida dan jaga sanitasi.',
        'img': 'grey_mold_70.jpg',
        'detail_img': 'grey_mold_71.jpg'
    },
    'gryllotalpa': {
        'title': 'Gryllotalpa (Orong-orong / Anjing Tanah)', 
        'desc': 'Hama tanah bertubuh kuat yang memotong akar tanaman muda dari dalam tanah secara langsung.', 
        'symptom': 'Tanaman stroberi mendadak rebah dan layu karena akar utamanya terpotong habis di dalam tanah.', 
        'control': '1. Gunakan umpan beracun khusus hama tanah (insektisida butiran).<br>2. Olah tanah secara intensif sebelum tanam.',
        'img': 'gryllotalpa_1.jpg',
        'detail_img': 'gryllotalpa_2.jpg'
    },
    'healthy': {
        'title': 'Healthy', 
        'desc': 'Kondisi normal tanaman yang tumbuh subur, hijau, dan sehat.', 
        'symptom': 'Daun hijau segar, bebas hama, dan tidak ada tanda bercak anomali.', 
        'control': '1. Pertahankan pemeliharaan rutin.',
        'img': 'healthy_15.jpg',
        'detail_img': 'healthy_16.jpg'
    },
    'late_blight': {
        'title': 'Late Blight (Hawar Daun Akhir)', 
        'desc': 'Penyakit oomycete destruktif yang menghancurkan jaringan daun dalam waktu singkat.', 
        'symptom': 'Bercak basah kehitaman yang merambat cepat ke seluruh bagian tanaman dalam hitungan hari.', 
        'control': '1. Semprot fungisida pencegah secara ketat saat musim hujan.',
        'img': 'late_blight_13.jpg',
        'detail_img': 'late_blight_14.jpg'
    },
    'leaf_deformation': {
        'title': 'Leaf Deformation', 
        'desc': 'Kelainan pertumbuhan bentuk daun yang tidak simetris.', 
        'symptom': 'Daun tumbuh melengkung, berkerut, atau asimetris.', 
        'control': '1. Cek serangan virus atau residu kimia herbisida.',
        'img': 'leaf_deformation_9.jpg',
        'detail_img': 'leaf_deformation_10.jpg'
    },
    'leaf_miners': {
        'title': 'Leaf Miners (Pengorok Daun)', 
        'desc': 'Larva lalat kecil yang hidup dan membuat terowongan di dalam daging daun.', 
        'symptom': 'Terdapat jalur liukan putih atau perak yang berbelok-belok di dalam helai daun.', 
        'control': '1. Gunakan insektisida translaminar.<br>2. Petik daun berterowongan parah.',
        'img': 'leaf_miners_13.jpg',
        'detail_img': 'leaf_miners_14.jpg'
    },
    'leaf_spot': {
        'title': 'Leaf Spot', 
        'desc': 'Penyakit bercak daun umum yang disebabkan oleh jamur Mycosphaerella fragariae.', 
        'symptom': 'Bercak ungu kecil berpusat terang di daun.', 
        'control': '1. Aplikasikan fungisida mancozeb.',
        'img': 'leaf_spot_13.jpg',
        'detail_img': 'leaf_spot_14.jpg'
    },
    'leaves_scorch': {
        'title': 'Leaves Scorch', 
        'desc': 'Variasi gejala daun hangus terbakar akibat stres lingkungan.', 
        'symptom': 'Tepian daun kering dan berwarna cokelat.', 
        'control': '1. Jaga suplai air.',
        'img': 'leaves_scorch_1.jpg',
        'detail_img': 'leaves_scorch_2.jpg'
    },
    'lichen': {
        'title': 'Lichen (Lumut Kerak)', 
        'desc': 'Simbiosis lumut kerak yang menempel pada batang tanaman tua.', 
        'symptom': 'Bercak kerak hijau keabuan menempel di kulit batang.', 
        'control': '1. Sikat pembersihan fisik secara lembut.',
        'img': 'lichen_17.jpg',
        'detail_img': 'lichen_18.jpg'
    },
    'loss_of_foliage_turgor': {
        'title': 'Loss of Foliage Turgor', 
        'desc': 'Kehilangan tekanan turgor sel daun.', 
        'symptom': 'Daun terkulai lemas akibat kekurangan air atau kerusakan akar.', 
        'control': '1. Lakukan penyiraman segera.',
        'img': 'loss_of_foliage_turgor_25.jpg',
        'detail_img': 'loss_of_foliage_turgor_26.jpg'
    },
    'marginal_leaf_necrosis': {
        'title': 'Marginal Leaf Necrosis', 
        'desc': 'Kematian jaringan di sepanjang tepi helai daun.', 
        'symptom': 'Pinggiran daun menghitam, mengering, dan mati.', 
        'control': '1. Perbaiki nutrisi kalium.',
        'img': 'marginal_leaf_necrosis_13.jpg',
        'detail_img': 'marginal_leaf_necrosis_14.jpg'
    },
    'mealybug': {
        'title': 'Mealybug (Kutu Tepung)', 
        'desc': 'Hama penghisap yang tubuhnya dilapisi serbuk lilin putih.', 
        'symptom': 'Koloni putih bertepung menempel di ketiak daun dan tangkai.', 
        'control': '1. Oleskan alkohol atau semprot sabun insektisida.',
        'img': 'mealybug_5.jpg',
        'detail_img': 'mealybug_6.jpg'
    },
    'mechanical_damage': {
        'title': 'Mechanical Damage', 
        'desc': 'Kerusakan fisik akibat gesekan alat atau hewan.', 
        'symptom': 'Luka sobek atau patah pada bagian tanaman.', 
        'control': '1. Penanganan budidaya yang hati-hati.',
        'img': 'mechanical_damage_33.jpg',
        'detail_img': 'mechanical_damage_34.jpg'
    },
    'monilia': {
        'title': 'Monilia', 
        'desc': 'Jamur penyebab busuk bunga dan buah.', 
        'symptom': 'Pembusukan cokelat pada organ bunga dan buah.', 
        'control': '1. Gunakan fungisida.',
        'img': 'monilia_11.jpg',
        'detail_img': 'monilia_12.jpg'
    },
    'mosaic_virus': {
        'title': 'Mosaic Virus', 
        'desc': 'Penyakit sistemik akibat infeksi virus mozaik.', 
        'symptom': 'Daun belang hijau muda-tua berpola mozaik dan kerdil.', 
        'control': '1. Basmi kutu vektor pembawa virus dan cabut tanaman sakit.',
        'img': 'mosaic_virus_6.jpg',
        'detail_img': 'mosaic_virus_7.jpg'
    },
    'northern_leaf_blight': {
        'title': 'Northern Leaf Blight', 
        'desc': 'Penyakit hawar daun berbentuk gelendong.', 
        'symptom': 'Lesi berbentuk kumparan memanjang pada daun.', 
        'control': '1. Semprot fungisida.',
        'img': 'northern_leaf_blight_9.jpg',
        'detail_img': 'northern_leaf_blight_10.jpg'
    },
    'nutrient_deficiency': {
        'title': 'Nutrient Deficiency', 
        'desc': 'Kekurangan unsur hara makro/mikro tanah.', 
        'symptom': 'Klorosis, daun menguning, dan pertumbuhan terhambat.', 
        'control': '1. Berikan pupuk NPK berimbang.',
        'img': 'nutrient_deficiency_10.jpg',
        'detail_img': 'nutrient_deficiency_11.jpg'
    },
    'pear_blister_mite': {
        'title': 'Pear Blister Mite', 
        'desc': 'Tungau pembentuk bintil.', 
        'symptom': 'Bintil-bintil menonjol pada permukaan daun.', 
        'control': '1. Gunakan akarisida.',
        'img': 'pear_blister_mite_2.jpg',
        'detail_img': 'pear_blister_mite_3.jpg'
    },
    'pest_damage': {
        'title': 'Pest Damage', 
        'desc': 'Kerusakan umum akibat berbagai hama.', 
        'symptom': 'Bekas gigitan dan kerusakan fisik.', 
        'control': '1. Kendalikan dengan pestisida yang sesuai.',
        'img': 'pest_damage_1.jpg',
        'detail_img': 'pest_damage_2.jpg'
    },
    'polypore': {
        'title': 'Polypore', 
        'desc': 'Jamur kayu pemecah lignin.', 
        'symptom': 'Tubuh buah jamur menempel di batang.', 
        'control': '1. Potong bagian kayu terinfeksi.',
        'img': 'polypore_2.jpg',
        'detail_img': 'polypore_3.jpg'
    },
    'powdery_mildew': {
        'title': 'Powdery Mildew', 
        'desc': 'Penyakit embun tepung umum.', 
        'symptom': 'Lapisan putih tepung di permukaan.', 
        'control': '1. Fungisida sulfur.',
        'img': 'powdery_mildew_2.jpg',
        'detail_img': 'powdery_mildew_3.jpg'
    },
    'powdery_mildew_fruit': {
        'title': 'Powdery Mildew Fruit', 
        'desc': 'Embun tepung yang menyerang buah stroberi.', 
        'symptom': 'Buah tertutup serbuk putih dan gagal matang.', 
        'control': '1. Semprot fungisida aman pangan.',
        'img': 'powdery_mildew_fruit_3.jpg',
        'detail_img': 'powdery_mildew_fruit_4.jpg'
    },
    'powdery_mildew_leaf': {
        'title': 'Powdery Mildew Leaf', 
        'desc': 'Penyakit tepung putih daun.', 
        'symptom': 'Serbuk putih menutupi daun dan menggulung.', 
        'control': '1. Fungisida sulfur atau minyak neem.',
        'img': 'powdery_mildew_leaf_395.jpg',
        'detail_img': 'powdery_mildew_leaf_396.jpg'
    },
    'rust': {
        'title': 'Rust (Karat Daun)', 
        'desc': 'Penyakit jamur karat.', 
        'symptom': 'Bintil oranye kecokelatan seperti serbuk karat di bawah daun.', 
        'control': '1. Aplikasikan fungisida kontak.',
        'img': 'rust_21.jpg',
        'detail_img': 'rust_22.jpg'
    },
    'scab': {
        'title': 'Scab (Kudis)', 
        'desc': 'Penyakit kudis pada permukaan jaringan.', 
        'symptom': 'Kudis keras kasar pada daun atau buah.', 
        'control': '1. Gunakan fungisida.',
        'img': 'scab_15.jpg',
        'detail_img': 'scab_16.jpg'
    },
    'scale': {
        'title': 'Scale (Kutu Perisai)', 
        'desc': 'Hama bertubuh keras dilindungi perisai lilin.', 
        'symptom': 'Bintik-bintik menonjol keras menempel di batang.', 
        'control': '1. Insektisida minyak hortikultura.',
        'img': 'scale_2.jpg',
        'detail_img': 'scale_3.jpg'
    },
    'shot_hole': {
        'title': 'Shot Hole', 
        'desc': 'Penyakit yang membuat daun berlubang seperti tertembak peluru.', 
        'symptom': 'Bercak daun yang bagian tengahnya copot meninggalkan lubang rapi.', 
        'control': '1. Semprot fungisida.',
        'img': 'shot_hole_1.jpg',
        'detail_img': 'shot_hole_2.jpg'
    },
    'shute': {
        'title': 'Shute', 
        'desc': 'Penyakit kerontokan daun massal.', 
        'symptom': 'Daun mengering dan gugur sebelum waktunya.', 
        'control': '1. Fungisida.',
        'img': 'shute_7.jpg',
        'detail_img': 'shute_8.jpg'
    },
    'slugs': {
        'title': 'Slugs (Siput Tanpa Cangkang / Keong)', 
        'desc': 'Hama moluskum yang aktif di malam hari memakan daun.', 
        'symptom': 'Daun compang-camping dengan jejak lendir mengkilap di sekitarnya.', 
        'control': '1. Taburkan umpan siput (moluskasida) di sekitar tanaman.',
        'img': 'slugs_13.jpg',
        'detail_img': 'slugs_14.jpg'
    },
    'slugs_caterpillars_effects': {
        'title': 'Slugs & Caterpillars Effects', 
        'desc': 'Dampak kerusakan gabungan hama pemakan daun.', 
        'symptom': 'Helaian daun rusak parah dan bolong-bolong.', 
        'control': '1. Pembersihan fisik & pestisida.',
        'img': 'slugs_caterpillars_effects_35.jpg',
        'detail_img': 'slugs_caterpillars_effects_36.jpg'
    },
    'sooty_mold': {
        'title': 'Sooty Mold (Embun Jelaga)', 
        'desc': 'Jamur hitam yang hidup dari cairan manis sisa sekresi kutu.', 
        'symptom': 'Lapisan hitam seperti beludru arang menutupi permukaan daun.', 
        'control': '1. Basmi kutu penghasil madu.',
        'img': 'sooty_mold_1.jpg',
        'detail_img': 'sooty_mold_2.jpg'
    },
    'spider_mite': {
        'title': 'Spider Mite (Tungau Laba-laba)', 
        'desc': 'Hama mikroskopis pengisap cairan sel.', 
        'symptom': 'Daun kusam berbintik kuning dan ada jaring halus.', 
        'control': '1. Akarisida & semprot air tekanan tinggi.',
        'img': 'spider_mite_7.jpg',
        'detail_img': 'spider_mite_8.jpg'
    },
    'thrips': {
        'title': 'Thrips', 
        'desc': 'Hama serangga kecil perusak jaringan.', 
        'symptom': 'Daun berwarna keperakan dan mengeriting.', 
        'control': '1. Insektisida nabati.',
        'img': 'thrips_1.jpg',
        'detail_img': 'thrips_2.jpg'
    },
    'tubercular_necrosis': {
        'title': 'Tubercular Necrosis', 
        'desc': 'Nekrosis bertuberkel pada jaringan.', 
        'symptom': 'Bintik mati jaringan bernomol.', 
        'control': '1. Fungisida.',
        'img': 'tubercular_necrosis_1.jpg',
        'detail_img': 'tubercular_necrosis_2.jpg'
    },
    'verticillium_wilt': {
        'title': 'Verticillium Wilt', 
        'desc': 'Penyakit layu pembuluh oleh jamur tanah Verticillium.', 
        'symptom': 'Tanaman layu mendadak, daun tua menguning dan mengering dari bawah.', 
        'control': '1. Sterilisasi tanah & rotasi tanaman.',
        'img': 'verticillium_wilt_3.jpg',
        'detail_img': 'verticillium_wilt_4.jpg'
    },
    'whitefly': {
        'title': 'Whitefly (Kutu Putih Kecil)', 
        'desc': 'Hama lalat putih kecil yang bergerombol di balik daun.', 
        'symptom': 'Serangga putih kecil beterbangan saat daun disentuh, daun menguning.', 
        'control': '1. Pasang perangkap kuning lekat.',
        'img': 'whitefly_1.jpg',
        'detail_img': 'whitefly_2.jpg'
    },
    'wilting': {
        'title': 'Wilting (Layu Umum)', 
        'desc': 'Gejala umum kehilangan kesegaran tanaman.', 
        'symptom': 'Seluruh bagian tanaman terkulai lemas.', 
        'control': '1. Perbaiki penyiraman & akar.',
        'img': 'wilting_5.jpg',
        'detail_img': 'wilting_6.jpg'
    },
    'wireworm': {
        'title': 'Wireworm (Ulat Kawat)', 
        'desc': 'Larva kumbang tanah berbadan keras.', 
        'symptom': 'Akar berlubang dan tanaman layu.', 
        'control': '1. Insektisida butiran tanah.',
        'img': 'wireworm_1.jpg',
        'detail_img': 'wireworm_2.jpg'
    },
    'wireworm_effects': {
        'title': 'Wireworm Effects', 
        'desc': 'Dampak kerusakan sistem perakaran oleh ulat kawat.', 
        'symptom': 'Perakaran keropos dan rusak.', 
        'control': '1. Pengolahan tanah.',
        'img': 'wireworm_effects_2.jpg',
        'detail_img': 'wireworm_effects_3.jpg'
    },
    'yellow_leaves': {
        'title': 'Yellow Leaves (Daun Menguning)', 
        'desc': 'Gejala umum klorosis daun.', 
        'symptom': 'Helai daun berubah warna menjadi kuning pucat.', 
        'control': '1. Cek pupuk Nitrogen.',
        'img': 'yellow_leaves_6.jpg',
        'detail_img': 'yellow_leaves_7.jpg'
    }
}

# Kamus Data Jenis-Jenis Stroberi dengan Penjelasan Lengkap
STRAWBERRY_TYPES = {
    'setaria': {
        'title': 'Stroberi California',
        'desc': 'Varietas stroberi yang paling populer dan banyak dibudidayakan di dataran tinggi Indonesia. Jenis ini memiliki kemampuan adaptasi yang sangat baik terhadap kondisi iklim tropis serta struktur tanah pegunungan, sehingga menjadi primadona bagi para petani lokal.',
        'symptom': 'Buah berukuran sedang hingga besar dengan bentuk kerucut sempurna dan warna merah cerah mengkilap. Memiliki tekstur daging buah yang padat, rasa manis yang dominan bercampur kesegaran asam alami, serta aroma buah yang sangat harum.',
        'control': '1. Cocok ditanam pada kisaran suhu sejuk 15-20 derajat Celsius dengan pencahayaan matahari penuh secara konsisten.<br>2. Lakukan pemupukan secara berkala menggunakan kompos organik serta penambahan unsur hara makro dan mikro yang seimbang.<br>3. Jaga kelembapan media tanam agar tetap stabil dan pastikan sistem drainase berjalan lancar guna mencegah risiko kebusukan akar.',
        'img': 'california.jpg',
        'detail_img': 'california1.jpg'
    },
    'sweet_charlie': {
        'title': 'Stroberi Sweet Charlie',
        'desc': 'Jenis stroberi unggulan asal Florida, Amerika Serikat, yang dikenal memiliki ketahanan tinggi terhadap fluktuasi cuaca hangat serta siklus masa panen yang relatif cepat dibandingkan varietas konvensional lainnya.',
        'symptom': 'Buah berbentuk agak pendek melebar dengan pangkal yang lebih besar, berwarna merah cerah terang, memiliki rasa yang sangat manis bahkan ketika belum terlalu matang merah, serta digemari karena ketahanan simpannya.',
        'control': '1. Membutuhkan sistem penyiraman teratur serta suplai pemupukan unsur Fosfor (P) yang optimal untuk merangsang pembungaan.<br>2. Lakukan pemangkasan daun tua secara rutin untuk memaksimalkan sirkulasi udara di sekitar pangkal tanaman.<br>3. Aplikasikan mulsa plastik hitam perak guna menjaga kebersihan buah dari kontak langsung dengan tanah lembap.',
        'img': 'carlie.jpg',
        'detail_img': 'carlie1.jpg'
    },
    'festival': {
        'title': 'Stroberi Festival',
        'desc': 'Varietas komersial berdaya hasil tinggi yang sering menjadi pilihan utama petani modern karena kekokohan struktur buahnya dan ketahanannya yang luar biasa terhadap proses pengiriman jarak jauh.',
        'symptom': 'Warna buah merah tua merata hingga ke bagian dalam, ukuran buah cenderung seragam dari petikan pertama hingga akhir, daging buah sangat padat, serta memiliki daya simpan buah yang jauh lebih lama di suhu ruang.',
        'control': '1. Pelihara jarak tanam antar bedengan agar pencahayaan matahari dapat merata ke seluruh tajuk tanaman.<br>2. Lakukan pengendalian hama pengisap secara preventif guna menjaga kebersihan dan kualitas kilau permukaan kulit buah.<br>3. Berikan pengairan tetes (drip irrigation) untuk efisiensi penyerapan nutrisi tanaman secara maksimal.',
        'img': 'festival.jpg',
        'detail_img': 'festival1.jpg'
    },
    'alba': {
        'title': 'Stroberi Alba',
        'desc': 'Varietas stroberi modern asal Italia yang sangat produktif dan dirancang khusus untuk menghasilkan buah berukuran besar dengan standar kualitas tinggi untuk pasaran ekspor maupun agrowisata.',
        'symptom': 'Karakteristik buah berbentuk memanjang (kerucut memanjang/long conical) dengan warna merah terang mengkilap yang sangat menarik. Buah tidak mudah berlubang di bagian tengah dan memiliki ketahanan tinggi terhadap keretakan akibat hujan.',
        'control': '1. Memerlukan media tanam yang gembur (porous) dengan sistem drainase yang sangat baik agar sistem perakaran tidak mudah membusuk.<br>2. Berikan naungan pelindung tambahan jika intensitas curah hujan di area budidaya terlalu ekstrem.<br>3. Lakukan pemantauan kesehatan daun secara berkala untuk mengantisipasi serangan jamur sekunder.',
        'img': 'alba.jpg',
        'detail_img': 'alba1.jpg'
    }
}

def load_model():
    model = models.googlenet(weights=None, aux_logits=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load('best_googlenet_strawberry_balanced.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

latest_result = None

@app.route('/', methods=['GET', 'POST'])
def index():
    global latest_result
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            image = Image.open(filepath).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                
            conf, predicted = torch.max(probabilities, 0)
            predicted_class = class_names[predicted.item()]
            confidence_score = round(conf.item() * 100, 2)
            all_probs = {class_names[i]: round(probabilities[i].item() * 100, 2) for i in range(len(class_names))}
            
            latest_result = {
                'prediction': predicted_class,
                'confidence': confidence_score,
                'probabilities': all_probs,
                'image_path': url_for('static', filename=f'uploads/{filename}')
            }
            return redirect(url_for('index'))
            
    res = latest_result
    latest_result = None
    return render_template('index.html', result=res)

@app.route('/info-penyakit')
def info_penyakit():
    page = request.args.get('page', 1, type=int)
    per_page = 6
    
    disease_items = list(DISEASE_INFO.items())
    total_items = len(disease_items)
    
    total_pages = (total_items + per_page - 1) // per_page
    
    start = (page - 1) * per_page
    end = start + per_page
    paginated_diseases = dict(disease_items[start:end])
    
    return render_template('info.html', 
                           diseases=paginated_diseases, 
                           current_page=page, 
                           total_pages=total_pages,
                           start_index=start + 1,
                           end_index=min(end, total_items),
                           total_items=total_items)

@app.route('/penyakit/<key>')
def detail_penyakit(key):
    disease = DISEASE_INFO.get(key)
    if not disease:
        return redirect(url_for('info_penyakit'))
    return render_template('detail_penyakit.html', disease=disease, key=key)

@app.route('/jenis-stroberi')
def jenis_stroberi():
    return render_template('jenis_stroberi.html', strawberry_types=STRAWBERRY_TYPES)

@app.route('/jenis-stroberi/<key>')
def detail_jenis(key):
    stype = STRAWBERRY_TYPES.get(key)
    if not stype:
        return redirect(url_for('jenis_stroberi'))
    return render_template('detail_jenis.html', stype=stype)

@app.route('/profil')
def profil():
    profil_putri = {
        'nama': 'Putri Regina Tambunan',
        'peran': 'Pencipta & Peneliti Aplikasi',
        'jurusan': 'Teknik Informatika - Fakultas Teknik UMA',
        'judul': 'Implementasi Deep Learning Arsitektur GoogLeNet untuk Deteksi Penyakit Daun Stroberi',
        'keahlian': ['Deep Learning', 'Image Processing', 'Web Developer', 'Python & PyTorch'],
        'kutipan': 'Teknologi kecerdasan buatan harus mampu memberikan solusi nyata dan solutif bagi sektor pertanian modern.',
        'foto': 'putri.jpeg'
    }
    return render_template('profil.html', profil=profil_putri)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)