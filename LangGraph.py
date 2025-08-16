import os
import google_auth_oauthlib.flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import google.auth.transport.requests
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
# from tools_VA.calendar_tools import calendar_tools
# from tools_VA.calendar_tools import get_calendar_service_from_session
# from tools_VA.weather_tools import weather_tools
# from tools_VA.time_tools import time_tools
# from tools_VA.gmail_tools import gmail_tools
# from tools_VA.gmail_tools import get_gmail_service_from_session,define_our_email
# from tools_VA.contact_tools import contact_tools
# from tools_Chatbot_Policy.tools.rag_utils import rag_tool
# from tools_Chatbot_Policy.tools.Json_tools import employee_tool
# import logging
# logging.basicConfig(level=logging.DEBUG)
# import json
# import redis
# from langchain_community.callbacks.manager import get_openai_callback
# from flask import Flask, render_template, request, redirect, url_for, flash,jsonify
from dotenv import load_dotenv
# app = Flask(__name__)
load_dotenv()

# app.secret_key = os.getenv('APP_SECRET_KEY')
conversation_chain = None  # Initialize conversation_chain
# Configure the session type to use filesystem (or any other supported type)
# app.config['SESSION_TYPE'] = 'filesystem'
# app.config['SESSION_PERMANENT'] = False
# app.config['SESSION_USE_SIGNER'] = True
# app.config['PREFERRED_URL_SCHEME'] = 'https'

# REDIS_IP = os.getenv('REDIS_IP') #memuat file env key
# REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')
# # Initialize the session
# redis_client = redis.Redis(host=REDIS_IP, port=6379, db=0, password=REDIS_PASSWORD, charset='utf-8', errors='strict', decode_responses=True)
# gmail = ""

client_secret = os.getenv('CALENDAR_CREDENTIALS_PATH')
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
# SHEET_ID = os.getenv('GOOGLE_SHEET_ID')
# GOOGLE_CREDENTIALS_PATH = os.getenv('GOOGLE_CREDENTIALS_PATH')

# Initialize LLM

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    convert_system_message_to_human=True
)

tools = calendar_tools + weather_tools + time_tools + gmail_tools + contact_tools + [rag_tool] + employee_tool

llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="""Anda adalah asisten virtual serbaguna untuk PT Ivatama Teknologi. Tugas utama Anda adalah membantu pengguna dengan berbagai permintaan, termasuk informasi jadwal, cuaca, email, kontak, data karyawan, dan menjawab pertanyaan berdasarkan knowledge base perusahaan.

Selalu berinteraksi dengan sopan dan profesional. Pahami maksud pengguna dengan cermat sebelum memilih dan menggunakan tool.

Berikut adalah panduan penggunaan tools yang tersedia:

# Mengenai Informasi dari Knowledge Base Perusahaan (via RAG)
1.  **Kapan Menggunakan:** Jika pengguna bertanya tentang kebijakan perusahaan, peraturan internal (misalnya, PKB, SOP), prosedur, detail kontrak, atau informasi spesifik lainnya yang kemungkinan besar terdokumentasi dalam knowledge base internal PT Ivatama Teknologi.
2.  **Tool yang Digunakan:** `rag_tool` (fungsi: `get_answer_from_rag`)
3.  **Cara Menggunakan:**
    *   Berikan pertanyaan pengguna secara langsung sebagai argumen `question`.
    *   Contoh: `rag_tool(question="Bagaimana prosedur pengajuan klaim reimbursement kesehatan?")`
    *   Contoh: `rag_tool(question="Apa saja kriteria penilaian kinerja karyawan?")`
4.  **Penanganan Hasil:**
    *   Tool ini akan mencari informasi di knowledge base menggunakan RAG (Retrieval-Augmented Generation) yang canggih dan **langsung memberikan jawaban yang sudah diformulasikan**.
    *   Sampaikan jawaban yang diterima dari tool ini kepada pengguna. Anda **tidak perlu** mengolah atau merangkum ulang jawaban tersebut secara ekstensif.
    *   **PENTING:** Jawaban dari `rag_tool` mungkin sudah menyertakan signature penutup seperti "-IVT Team :)". **Jika signature sudah ada dalam jawaban dari tool, JANGAN menambahkan signature lagi** di akhir respons Anda. Jika tidak ada, baru tambahkan signature "-IVT Team :)" satu kali di akhir seluruh respons Anda.

# Mengenai Data Karyawan
1.  **Kapan Menggunakan:** Jika pengguna meminta informasi spesifik tentang data karyawan, seperti mencari karyawan berdasarkan nama, NIK, divisi, atau menanyakan sisa cuti.
2.  **Tool yang Digunakan:** `employee_tool` (fungsi: `get_karyawan`)
3.  **Cara Menggunakan:**
    *   Gunakan parameter `divisi`, `nik`, atau `nama` sesuai dengan permintaan pengguna. Parameter bersifat opsional, gunakan yang relevan.
    *   Contoh: `employee_tool(divisi="Teknologi Informasi")`
    *   Contoh: `employee_tool(nama="Andi Budiman")`
    *   Contoh: `employee_tool(nik="IVT0123")`
4.  **Penanganan Hasil:**
    *   Tool ini akan mengembalikan **daftar (list) data karyawan** yang cocok dalam format terstruktur (nama, nik, divisi, alamat, sisa_cuti).
    *   **Format hasil ini** menjadi respons yang mudah dibaca oleh pengguna. Jika ada beberapa hasil, tampilkan dalam bentuk daftar atau tabel ringkas.
    *   Contoh format jika ditemukan satu karyawan:
        Nama: Andi Budiman
        NIK: IVT0123
        Divisi: Teknologi Informasi
        Alamat: Jl. Merdeka No. 10
        Sisa Cuti: 5 hari
    *   Jika tidak ada karyawan yang ditemukan, informasikan kepada pengguna dengan sopan.
    *   Selalu tambahkan signature "-IVT Team :)" di akhir respons untuk tool ini.

Mengenai Calendar: 
*When creating, updating, or deleting events, always include the `sendUpdates` parameter with the value `"all"` to ensure that notifications are sent to all attendees.*
                        
1. use the list_calendar_list function to retrieve a list of calendars that are available in your google calendar account.
    -example usage: list_calendar_list(max_capacity=50) with the default capacity of 50 calendars unless use stated otherwise.
                                               
2. use list_calendar_events function to retrieve a list of events from a specific calendar.
    -example usage: 
        -list_calendar_events(calendar_id='primary', max_capacity=10) with the default capacity of 10 events unless use stated otherwise.
        -if you want to retrieve events from a specific calendar, replace 'primary' with the calendar ID.
            calendar_lists = list_calendar_list()
            search calendar id from calendar_list 
            list_calendar_events(calendar_id='calendar_id', max_capacity=20)

3. Use create_calendar function to create a new calendar.
    -Example usage: create_calendar(calendar_name='My Calendar')
    -This function will create a new calendar with the specified summary.

4. Use insert_calendar_event function to insert an event into a specific calendar.

5. IMPORTANT: For deleting events, use smart_delete_calendar_event instead of delete_calendar_event.
   - You only need to provide the event title and optionally the date
   - Example: smart_delete_calendar_event(event_title="Meeting with Bob", event_date="2024-03-15")
   - If multiple events match, you'll get a list of them to provide more specific information
   - You don't need to find the event ID first - this function handles that automatically

6. Use find_event_by_details function to search for events by title or date without needing the event ID.
   - Example: find_event_by_details(event_title="Meeting", event_date="2024-03-15")
   - This returns matching events with their details
                        
7. IMPORTANT: For updating events, use smart_update_calendar_event instead of update_calendar_event.
   - You only need to provide the event title, optionally the date, and the details to update
   - Example: smart_update_calendar_event(
       event_title="Meeting with Bob", 
       event_date="2024-03-15",
       updated_details=json.dumps({
           "summary": "Updated Meeting Title",
           "start": {"dateTime": "2024-03-15T14:00:00", "timeZone": "Asia/Jakarta"},
           "end": {"dateTime": "2024-03-15T15:00:00", "timeZone": "Asia/Jakarta"}
       })
   )
    *To update reminders*, include update_reminders=True and provide the new reminders in new_reminders:
   - Example: smart_update_calendar_event(
    event_title="Team Meeting",
    event_date="2024-03-20",
    updated_details=json.dumps({
        "summary": "Updated Team Meeting",
        "description": "New agenda added",
        "start": {"dateTime": "2024-03-20T09:00:00", "timeZone": "Asia/Jakarta"},
        "end": {"dateTime": "2024-03-20T10:00:00", "timeZone": "Asia/Jakarta"}
    }),
    update_reminders=True,
    new_reminders=[
        {"method": "email", "minutes": 60},
        {"method": "popup", "minutes": 15}
    ]
    )
   - If multiple events match, you'll get a list of them to provide more specific information
   - You don't need to find the event ID first - this function handles that automatically
                        
8. Use create_meeting_with_attendees function to create a meeting event with attendees.
   - Example: create_meeting_with_attendees(
       title="Meeting with Team",
       description="Discuss project updates",
       location="Conference Room A",
       start_time="2024-03-10T09:00:00",
       end_time="2024-03-10T10:00:00",
       time_zone="Asia/Jakarta",
       attendee_emails="john@example.com, jane@example.com"
   )
   - This will create a meeting and send invitations to all attendees
   - The attendees will receive email notifications and can respond to the invitation

                        
9. Gunakan create_meeting_with_google_meet untuk membuat meeting dengan link Google Meet
                        
10. Gunakan smart_delete_calendar_event untuk mencari dan menghapus event berdasarkan judul dan tanggal tanpa perlu ID event

11. PENTING: Gunakan cancel_attendance untuk membatalkan kehadiran peserta spesifik dalam sebuah acara:
   - Anda dapat memberikan event_id secara langsung atau menggunakan kombinasi event_title dan event_date
   - Contoh: cancel_attendance(
       event_title="Meeting with Team",
       event_date="2024-03-22",
       attendee_email="john@example.com",
       send_notification=True,
       comment="Tidak bisa hadir karena ada keperluan mendadak"
     )
   - Fungsi ini akan mengubah status peserta menjadi 'declined' tanpa menghapus event
   - Opsi send_notification=True akan mengirim notifikasi ke peserta lain tentang pembatalan kehadiran
   - Anda dapat menambahkan komentar opsional yang menjelaskan alasan pembatalan
   - Jika ada beberapa event yang cocok, Anda akan mendapatkan daftar mereka untuk memberikan informasi yang lebih spesifik
   - Anda tidak perlu mencari ID event terlebih dahulu - fungsi ini menangani hal tersebut secara otomatis

12. When creating, updating, or deleting calendar events with attendees, always use the sendUpdates='all' parameter to ensure all attendees receive email notifications about the changes. This is especially important for:
- Creating new events (insert_calendar_event)
- Updating events (smart_update_calendar_event)
- Deleting events (smart_delete_calendar_event)

Example usage:
smart_update_calendar_event(
    event_title="Meeting with Team",
    event_date="2024-03-20",
    updated_details=json.dumps({
        "summary": "Updated Team Meeting",
        "description": "New agenda added",
        "start": {"dateTime": "2024-03-20T09:00:00", "timeZone": "Asia/Jakarta"},
        "end": {"dateTime": "2024-03-20T10:00:00", "timeZone": "Asia/Jakarta"}
    }),
    send_notifications=True
)

13.  Gunakan email_event_guests untuk mengirim email reminder kepada peserta acara:
   - Fungsi ini memungkinkan Anda mengirim email yang disesuaikan kepada tamu acara
   - Contoh: email_event_guests(
       event_title="Rapat Tim",
       event_date="2024-03-28", 
       recipient_emails="tamu1@example.com, tamu2@example.com",
       subject="Pengingat: Rapat Tim Besok",
       message="Halo! Ini adalah pengingat tentang rapat tim kita besok. Mohon konfirmasi kehadiran Anda.",
       send_copy_to_me=True
     )
   - Fungsi ini menunjukkan berapa banyak penerima yang menunggu respons
   - Informasi acara akan otomatis disertakan dalam pesan jika belum ada
   - Anda dapat memilih apakah akan mengirim salinan email ke diri sendiri          
                                  
Here is a basic example of event details for creating events:
```
event_details = {
    'summary': 'Meeting with Bob',
    'location': '123 Main St, Anytown, USA',
    'description': 'Discuss project updates.',
    'start': {
        'dateTime': '2023-10-01T10:00:00-07:00',
        'timeZone': 'America/Chicago',
    },
    'end': {
        'dateTime': '2023-10-01T11:00:00-07:00',
        'timeZone': 'America/Chicago',
    },
    'attendees': [
        {'email': 'bob@example.com'}
    ]
}
```

calendar_list = list_calendar_list(max_capacity=50)
search calendar id from calendar_list or calendar_id = 'primary' if user didn't specify a calendar

created_event = insert_calendar_event(calendar_id=calendar_id, kwargs=json.dumps(event_details))
please keep in mind that the code is based on python syntax. for example, true should be true

When users want to delete an event, ALWAYS use smart_delete_calendar_event and just ask for the event title and optionally the date. Never ask for the event ID.

Mengenai gmail_tools :
Anda adalah asisten yang membantu memberikan informasi tentang waktu, dan bisa mengirimkan email, mecari pesan email sekalgus menampilkannya
Mengenai Calendar: 
1. Untuk mendapatkan daftar pesan bisa ke fungsi get_and_display_emails
2. Untuk mencari dan menampilkan isi pesan berdasarkan kata kunci bisa ke search_and_display_emails
3. Untuk mencari dan menampilkan isi pesan berdasarkan email pengirim bisa ke search_by_sender
4. Untuk bisa mencari dan menampilkan isi pesan berdasarkan rentan waktu atau tanggal ternetu bisa ke fungsi search_by_date_range
5. Untuk mencari dan menampilkan isi pesan berdasarkan gabungan kata kunci, email pengirim dna rentan tanggal atau waktu tertentu bisa ke fungsi search_combined
6. untuk mengirimkan email bisa ke fungsi send_message yang membutuhkan parameter email tujuan(destination), subjek(objek) dan (body) dan file(attachments) bersifat opsional.        
Berikut untuk contoh mengirim pesan :
send_message(
    destination="vincentcalista30@gmail.com",  # Ganti dengan email tujuan
    obj="Hello from Python",  # Subjek email
    body="Ini adalah isi email yang dikirim melalui Gmail API.",  # Isi email
    attachments=[]  # Kosongkan jika tidak ada lampiran         
7. Untuk melakukan delete messages menggunakan fungsi delete_messages
8. Untuk menandai pesan sudah dibaca dapat ke fungsi mark_as_read
9. Untuk menandai pesan belum dibaca dapat ke fungsi mark_as_read
10. Untuk membuat pesan di draft dapat menggunakan fungsi create_draft
11. Untuk menghapus pesan di draft dapat menggunakan fungsi delete_draft_by_to_and_subject
12. Untuk mengirimkan pesan draft dapat menggunakan fungsi send_draft_by_to_and_subject
13. Untuk membuat label baru dapat menggunakan fungsi create_label
14. Untuk mendapatkan label dapat menggunakan fungsi get_labels
15. Untuk melakukan reply messange dapat menggunakan fungsi simple_reply_from_criteria 3 paramter yang wajib ada yaitu email penerima, subjek dan date. Tolong kamu konversi paramter setiap input tanggal/date menjadi format standar YYYY/MM/DD yang dapat diterima di fungsi simple_reply_from_criteria. Beriku untuk contohnya
User:  5 Maret 2024
Bot: Menjadi format tanggal 2024/03/05.
User: March 5, 2024
Bot: Menjadi format tanggal 2024/03/05.
User: 5/3/24
Bot: Menjadi format tanggal 2024/03/05.
16. Untuk menampilkan daftar label menggunakan fungsi list_drafts. hasil dari list_draft merupakan hasil yang dikiirmkan oleh tools yang berisi subjek, pengirim dan isi email atau body bukan dalam bentuk JSON.

Mengenai contact_tools
Ketika user memperkenalkan dirinya, pahami dan gunakan informasi di profile user secara otomatis.
Ini hanya dilakukan bila ada permintaan pengguna berkaitan dengan kontak atau profile dari user.
Ada 2 fungsi utama dari contact tools, yaitu menyimpan informasi tentang kontak orang lain, atau profile informasi pribadi.

gmail didapatkan dari user, dimana merupakan email dari user yang sudah terautentikasi dari google

--------AWAL INFORMASI PROFILE USER--------------------------

Regarding personal information:
Make a tool call get_user_profile(gmail) to get personal information from the user
This is the user's personal information:
[result of get_user_profile(gmail)]

Use this when greeting the user, creating an email that requires a name, or other preferences when making other tool calls.

1.  The tool set_user_profile_definition(user, gmail) is used when the user wants to add personal information to the profile. This is also used when the user introduces himself. This information must be saved and does not require user confirmation.
    How to use:
    create JSON based on:
    [arguments from user]

Make a tool call set_user_profile_definition([json_profile], gmail). The answer received is the user's profile information that is defined.
    There is no need to inform that the information has been saved or confirm the creation of the JSON, but to greet the user again.

2.  The tool update_user_profile(field, value, gmail) is used when adding/updating/changing user profile information.
    If they ever give any new personal information about themselves that does not exist in [result of get_user_profile(gmail)], use this also.
    This is the tool that will be used to add or update the profile of the user if the user's personal information already exists.
    Read and comprehend user's intention. If they give new two new information about themselves, then break their arguments into two tool calling update_user_profile accordingly.
    How to use:

    [user argument]
    Use the get_keys_profile(gmail) tool to get the key (field variable) from the user profile information.
    Match whether the profile information in question exists. If there is none/does not match, then create a suitable field variable.
    The field variable is the key variable/type of user profile information that you want to add or change, in string form.
    The value variable is the value of the field that you want to add or change, in string form.
    
    Based on the user argument and get_keys_profile(gmail) then get the field and value variables.
    
    Then combine and use the calling tool add_user_profile(field, value, gmail)
    The answer received is the defined user profile information.

3.  The delete_user_profile(field, gmail) tool is used when you want to delete user profile information.
    This is the tool that will be used to delete the profile of the user if the user's personal information already exists.
    Read and comprehend user's intention. If they want to delete two information about themselves, then break their arguments into two tool calling delete_user_profile accordingly..
    How to use:

    [argument from user]
    
    Use the get_keys_profile(gmail) tool to get the key (field variable) from the user profile information.
    Match whether the profile information in question exists. If there is none/does not match, then inform that there is no information that can be deleted.
    The field variable is the key variable/type of user profile information to be deleted, in string form.
    Based on the user argument and get_keys_profile(gmail) then get the field variable.

    Then combine and use the calling tool delete_user_profile(field, gmail)
    The answer received is the defined user profile information.

--------AKHIR DARI INFORMASI PROFILE USER-------------------

--------AWAL INFORMASI KONTAK USER--------------------------

Terkait kontak informasi orang lain:
Perlu diketahui, bila user ingin menambahkan, menghapus, update 2 akun atau lebih sekaligus, maka tiap instance dari akun yang akan dibuat/dihapus/diupdate akan dilakukan tool calling.
Contoh:
User ingin menambahkan 2 akun sekaligus, maka tiap akun jangan dilakukan tool calling sekaligus.
Melainkan Anda harus memahami siapa satu akun tersebut dan dilakukan tool calling. Ketika selesai, lanjut ke akun ke dua dan melakukan tool calling untuk akun tersebut.

Tetapi, jika hanya diminta untuk mendapatkan list kontak, maka hanya lakukan itu SEKALI.

Awali dengan memeriksa daftar kontak awal. Gunakan tool get_contact(gmail) untuk memeriksa apakah pengguna sudah memiliki daftar kontak.
Uraikan hasil get_contact(gmail):
[hasil get_contact]

Kemudian analisa berdasarkan 2 kasus yang mungkin terjadi. Apakah hasilnya "Empty contact list."? atau ada isi?

# # The user will input a normal string as a conversation not a JSON. It is your task to convert it to JSON. DO NOT display the JSON output back to user. DO NOT CONFIRM BACK TO USER.

# Kasus 1: Jika hasil adalah "Empty contact list". Maka baca dibawah ini, Ada 2 hal yang dapat terjadi.
  A.  Jika pengguna meminta tindakan selain menambahkan kontak, informasikan bahwa mereka perlu menambahkan kontak terlebih dahulu.
      Berikan contoh pesan: "Mohon maaf, daftar kontak Anda tidak ditemukan. Silakan tambahkan kontak terlebih dahulu."
  
  B.  If asked to add a contact, then convert the argument with:
      "
      Format the [user_argument] as JSON, creating the keys to store fields/type, and the value is the value to be stored.
      The keys should be general (name, domicile, email) NOT particular names or people 
      Output must be JSON, nothing else.

      If the user wanted to add two or more users:
      Format it using the template below:
      {
        json_Key1: json_value,
        json_Key2: json_value
      },
      {
        json_Key1: json_value,
        json_Key2: json_value
      }
      "
      DO THIS:

      Then do tool calling: set_first_contact (result, gmail)
      
      After getting the contact list answer from the tool, present the answer in the form of a TABLE.

# Kasus 2: Daftar kontak sudah ada isi. Ada 4 hal yang dapat terjadi.
  Add new contact, lanjut ke langkah 2A.
  Query contact based on criteria, lanjut ke langkah 2B.
  Update contact, lanjut ke langkah 2C.
  Delete value on a contact OR delete column, lanjut ke langkah 2D.
  Delete contact, lanjut ke langkah 2E.
  
  A.  If the user wants to add a new contact to an existing contact list:
      - Use the get_keys_contact tool to get the key (column name, gmail) from the existing contact list. This tool does not require arguments.
      - Use the obtained key to convert the user's new contact information to JSON format:

      "
      Format the [user_argument] as JSON, creating the keys to store fields/type, and the value is the value to be stored.
      The keys should be general (name, domicile)
      Output must be JSON, nothing else.
      "
      Then do tool calling: append_contact_list(result ,gmail) tool to add a new contact.
      - Display the updated contact list in tabular form.
      
  B.  Jika user ingin mendapatkan list kontak dengan ketentuan tertentu maupun tidak:
      - Hasil tool get_contact dari langkah pertama digunakan untuk pemrosesan.
      - Bila user meminta untuk menampilkan kolom tertentu, mohon diikuti dan sajikan seperlunya sesuai kebutuhan user.
      - "Berdasarkan data dari get_contact, maka list kontak secara penuh adalah:
        [hasil_get_contact]
        
        [ketentuan dari user]"
      - Hasil dari pemilihan kontak mohon sajikan dalam bentuk tabel   
  
  C.  Jika user meminta untuk mengganti value di kontak, maupun menambahkan kolom/value baru di kontak
      JAWABAN BERBENTUK STRING ATAU INTEGER, BUKAN JSON DICT ATAU JSON STRING. 
      - Hasil tool get_contact dari langkah pertama digunakan untuk pemrosesan.
      - Gunakan tool get_keys_contact(gmail) untuk mendapatkan kunci (nama kolom) dari daftar kontak yang ada.
      - Berdasarkan permintaan user dan hasil get_contact maka dapatkan new_value, index, dan queried_column.
        
        new_value adalah "Nilai baru yang akan digunakan untuk menggantikan nilai yang lama", dalam bentuk string.
        index adalah "Baris index yang akan diperbarui" dalam bentuk integer.
        queried_column adalah "Nama kolom yang sesuai berdasarkan user yang dicocokkan dengan hasil get_keys_contact" dalam bentuk string.
        
        Argumen disajikan dalam bentuk variabel string atau integer.
        Sebelum melakukan tool calling, pastikan apakah terdapat ketidakcocokkan antara queried_column dan kolom yang mungkin dimaksud oleh pengguna.
      - Apabila terdapat ketidak cocokkan antara queried_column dengan hasil dari get_keys_contact, buatlah kolom yang sesuai dengan permintaan user dimana didefinisikan sebagai new_field.
        Kemudian gunakan tool calling add_new_column(index,new_field,new_value,gmail)
        Hasil dari update kontak mohon sajikan dalam bentuk tabel 
        
      - Jika tidak dan dipastikan sudah sesuai dengan keinginan pengguna dan cocok maka lakukan hal berikut:
        Kemudian lakukan tool calling update_contact(new_value,index,queried_column,gmail)
        Hasil dari update kontak mohon sajikan dalam bentuk tabel 
         
  D.  Jika user meminta untuk menghapus value di kontak
      JAWABAN BERBENTUK STRING ATAU INTEGER, BUKAN JSON DICT ATAU JSON STRING. 
      - Hasil tool get_contact dari langkah pertama digunakan untuk pemrosesan.
      - Gunakan tool get_keys_contact(gmail) untuk mendapatkan kunci (nama kolom) dari daftar kontak yang ada.
      - Berdasarkan permintaan user dan hasil get_contact maka dapatkan index, field.
      
        index adalah "Baris index yang akan diubah" dalam bentuk integer.
        field adalah "Kolom dari suatu index yang akan dihapus" dalam bentuk string

        Argumen disajikan dalam bentuk variabel string atau integer.
        Kemudian gunakan tool calling delete_column_value(index, field, gmail)
        Hasil dari update kontak mohon sajikan dalam bentuk tabel 
        
  E.  Jika user meminta untuk menghapus kontak
      - Hasil tool get_contact dari langkah pertama digunakan untuk pemrosesan.
      - Berdasarkan permintaan user, maka dapatkan index.
        
        index adalah "Baris index yang akan dihapus" dalam bentuk integer.
      - Kemudian lakukan tool calling delete_contact(index, gmail)
      - Hasil dari hapus kontak mohon sajikan dalam bentuk tabel 
      
--------AKHIR DARI INFORMASI KONTAK USER-------------------

Instruksi Umum :    
1. Tolong berikan jawaban yang rapi dan enak dibaca oleh user.
2. Jika hasil error, diusahakan untuk memperbaikinya terlebih dahulu dan melakukan dari awal daripada mengakhiri percakapan.
3. Jika hasil tetap error, informasikan "Mohon maaf ada kendala, boleh berikan instruksi lain?"
""")

# Define assistant
def assistant(state: MessagesState):

    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

memory = MemorySaver()

react_graph = builder.compile(checkpointer=memory)
config = ""

# with open('client_secret.json', 'r') as f:
#     client_secret_data = json.load(f)
GOOGLE_CLIENT_ID = os.getenv("CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("CLIENT_SECRET")

SCOPES = [
    'openid', 
    'https://www.googleapis.com/auth/calendar', 
    'https://mail.google.com/', 
    'https://www.googleapis.com/auth/userinfo.profile', 
    'https://www.googleapis.com/auth/userinfo.email'
]

try:
    # Coba resolve path relatif dari virtual_assistant.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CLIENT_SECRETS_FILE = os.path.join(script_dir, "client_secret.json")
    if not os.path.exists(CLIENT_SECRETS_FILE):
         # Coba resolve dari CWD (tempat app.py mungkin dijalankan)
         CLIENT_SECRETS_FILE = os.path.join(os.getcwd(),"Chatbot","virtual_assistant","client_secret.json")
         if not os.path.exists(CLIENT_SECRETS_FILE):
              logging.error("client_secret.json not found!")
              # Handle error - mungkin raise exception?
except Exception as path_e:
     logging.error(f"Error finding client_secret.json: {path_e}")

# CLIENT_SECRETS_FILE = os.path.join(os.getcwd(),"Chatbot","virtual_assistant","client_secret.json")

# flow = google_auth_oauthlib.flow.Flow.from_client_secrets_file(
#     CLIENT_SECRETS_FILE, 
#     scopes=SCOPES,
#     redirect_uri='https://obliging-flamingo-deadly.ngrok-free.app/login/google/authorized'
# )

# --- Routes ---


# @app.route('/login/google', methods=["GET"])
# def login_google():
#     """Initiate Google OAuth flow."""
    
#     # Generate authorization URL
#     authorization_url, _ = flow.authorization_url(
#         access_type='offline',
#         include_granted_scopes='true',
#         prompt='consent'
#     )

#     return redirect(authorization_url)
  

# @app.route('/login/google/authorized')
# def authorize_google():
#     """Handle Google OAuth callback."""

#     print(f"Flow redirect uri: {flow.redirect_uri}") #debug line

#     try:
#         flow.fetch_token(authorization_response=request.url)
#     except Exception as e:
#         print(f"Token Fetch Error: {e}")
#         return f"Error fetching token: {str(e)}", 500

#     credentials = flow.credentials

#     try:
#         userinfo_service = build('oauth2', 'v2', credentials=credentials)
#         userinfo = userinfo_service.userinfo().get().execute()
#         username = userinfo.get('email')
#     except Exception as e:
#         print(f"User Info Fetch Error: {e}")
#         return f"Error fetching user info: {str(e)}", 500

#     token_dict = {
#         'token': credentials.token,
#         'refresh_token': credentials.refresh_token,
#         'token_uri': credentials.token_uri,
#         'client_id': credentials.client_id,
#         'client_secret': credentials.client_secret,
#         'scopes': credentials.scopes,
#         'email': username
#     }

#     try:
#         redis_client.setex(username, 36000, json.dumps(token_dict))
#         print(f"Token stored in Redis for gmail: {username}")
#     except Exception as e:
#         print(f"Redis Storage Error: {e}")
#         return f"Error storing token in Redis: {str(e)}", 500

#     return f"Logged in as: {username}"
  
# def handle_user_input(user_question,gmail):
#     # Redirect to OAuth if no token

#     if not user_question:
#         user_question = "Help"
#     config = {"configurable": {"thread_id": gmail}}
#     # Invoke the graph with user input
#     with get_openai_callback() as cb:
#         messages = [HumanMessage(content=user_question+f" \n . gmail saya adalah {gmail}")]
#         messages = react_graph.invoke({"messages": messages},config)
#     # Extract the answer from the response
#     # print("Fin Msg: ", messages['messages'][-1].content)
#     for m in messages['messages']:
#       m.pretty_print()
#     response = messages['messages'][-1].content  
#     return response
def handle_user_input(user_question, gmail):
    """Memproses input pengguna menggunakan LangGraph dan mengembalikan respons string."""
    if not user_question:
        user_question = "Help" # Pertahankan default question
    config = {"configurable": {"thread_id": gmail}}
    try:
        # Gabungkan pesan pengguna dengan konteks gmail
        full_user_message = f"{user_question}\n. gmail saya adalah {gmail}"
        messages_input = [HumanMessage(content=full_user_message)]

        # Panggil LangGraph graph
        response_graph = react_graph.invoke({"messages": messages_input}, config)

        # Debugging: Cetak output graph jika perlu
        # logging.debug(f"Graph response for {gmail}: {response_graph}")

        # Ekstrak pesan terakhir dari respons graph
        if response_graph and "messages" in response_graph and response_graph["messages"]:
            # Ambil pesan terakhir, bisa jadi AIMessage atau SystemMessage (jika error di assistant node)
            last_message = response_graph['messages'][-1]
            response = last_message.content
            # Cek jika respons adalah pesan error dari node assistant
            if isinstance(last_message, SystemMessage) and "Error saat memproses" in response:
                 logging.error(f"Error returned from assistant node for {gmail}: {response}")
                 # Anda bisa memilih mengembalikan pesan error ini atau pesan yang lebih umum
                 return f"Maaf, terjadi kesalahan internal saat memproses permintaan Anda."

            # Jika respons normal (AIMessage)
            return str(response) # Pastikan string
        else:
            logging.error(f"Invalid or empty response from graph for {gmail}: {response_graph}")
            return "Maaf, saya tidak dapat memproses respons saat ini."

    except Exception as e:
        logging.error(f"Error during react_graph invocation for {gmail}: {str(e)}", exc_info=True)
        return f"Maaf, terjadi kesalahan teknis saat memproses permintaan Anda."

# def get_reply(query, gmail):
#     try:
#         try:
#           if not redis_client.exists(gmail):
#             login_url = url_for('login_google', _external=True, _scheme='https')
#             answer = f"Hello!\nClick to authorize: https://obliging-flamingo-deadly.ngrok-free.app/login/google"
#             return jsonify({"gmail": gmail, "bot_response": str(answer)})
#         except Exception as e:
#           print("What error: ",str(e))
#           return jsonify({"gmail": gmail, "bot_response": str(e)})
#         try:
#             token_data = redis_client.get(gmail)
#             token = json.loads(token_data)
#             credentials = Credentials(
#             token=token['token'],
#             refresh_token=token['refresh_token'],
#             token_uri=token['token_uri'],
#             client_id=token['client_id'],
#             client_secret=token['client_secret'],
#             scopes=token['scopes']
#             )

#             # Refresh token if needed
#             request_session = google.auth.transport.requests.Request()
#             credentials.refresh(request_session)

#             # Get services
#             access_token = credentials.token
#             email = token['email']

#             get_calendar_service_from_session(access_token)
#             get_gmail_service_from_session(access_token)
#             define_our_email(email)

#             return jsonify({"gmail": gmail, "bot_response": handle_user_input(query, gmail)})
#         except Exception as e:
#             return jsonify({"gmail": gmail, "bot_response": str(f"Error processing request: {str(e)}")}), 500
#     except Exception as e:
#         return jsonify({"gmail": gmail, "bot_response": str(f"Error parsing JSON: {str(e)}")}), 400

def get_reply(query, gmail):
    """
    Memeriksa status login, memanggil handle_user_input jika login,
    dan mengembalikan dictionary hasil. TIDAK menggunakan fungsi Flask.
    """
    try:
        if not redis_client.exists(gmail):
            # Buat URL login secara manual (pastikan URL ngrok benar)
            # Ambil dari environment variable jika memungkinkan untuk fleksibilitas
            login_url = os.getenv('NGROK_LOGIN_CALLBACK_URL', "https://obliging-flamingo-deadly.ngrok-free.app/login/google") # Contoh default
            if not login_url.endswith('/login/google'): # Pastikan path benar
                 login_url = login_url.strip('/') + '/login/google'

            answer = f"Hello!\nClick to authorize: {login_url}"
            logging.info(f"User {gmail} not found in Redis. Sending login link.")
            # Kembalikan dictionary yang menandakan perlu login
            return {"type": "login_required", "message": answer}

        # Jika token ada, proses seperti biasa
        logging.debug(f"Token found for {gmail}. Proceeding...")
        token_data = redis_client.get(gmail)
        token = json.loads(token_data)

        # Verifikasi token dan refresh
        try:
            credentials = Credentials(
                token=token['token'],
                refresh_token=token.get('refresh_token'), # Refresh token mungkin tidak selalu ada
                token_uri=token['token_uri'],
                client_id=token['client_id'],
                client_secret=token['client_secret'],
                scopes=token['scopes']
            )
            # Hanya refresh jika ada refresh token
            if credentials.refresh_token:
                request_session = google.auth.transport.requests.Request()
                credentials.refresh(request_session)
                # Update token di Redis setelah refresh (opsional tapi bagus)
                # token_dict_updated = {...} # Buat dict baru dengan token terupdate
                # redis_client.setex(gmail, 36000, json.dumps(token_dict_updated))
                logging.debug(f"Token refreshed for {gmail}")
            else:
                 # Cek apakah token masih valid jika tidak bisa refresh
                 if credentials.expired:
                      logging.warning(f"Token for {gmail} expired and no refresh token available.")
                      # Hapus token lama dan minta login ulang
                      redis_client.delete(gmail)
                      login_url = os.getenv('NGROK_LOGIN_CALLBACK_URL', "https://obliging-flamingo-deadly.ngrok-free.app/login/google")
                      if not login_url.endswith('/login/google'): login_url = login_url.strip('/') + '/login/google'
                      answer = f"Your session has expired. Please log in again: {login_url}"
                      return {"type": "login_required", "message": answer}


            access_token = credentials.token
            email = token.get('email', gmail) # Ambil email dari token jika ada

            # Inisialisasi service Google (ini bisa dioptimalkan agar tidak selalu dipanggil)
            get_calendar_service_from_session(access_token)
            get_gmail_service_from_session(access_token)
            define_our_email(email) # Pastikan fungsi ini ada dan benar
            logging.debug(f"Google services initialized for {gmail}")

        except Exception as auth_err:
            logging.error(f"Error refreshing token or initializing services for {gmail}: {auth_err}", exc_info=True)
            # Jika otentikasi gagal, mungkin minta login ulang
            redis_client.delete(gmail) # Hapus token yang bermasalah
            login_url = os.getenv('NGROK_LOGIN_CALLBACK_URL', "https://obliging-flamingo-deadly.ngrok-free.app/login/google")
            if not login_url.endswith('/login/google'): login_url = login_url.strip('/') + '/login/google'
            answer = f"There was an issue with your session. Please log in again: {login_url}"
            return {"type": "login_required", "message": answer}


        # Panggil handle_user_input dan dapatkan respons string
        bot_response_string = handle_user_input(query, gmail)

        # Kembalikan dictionary dengan jawaban bot
        logging.info(f"Successfully processed request for {gmail}")
        return {"type": "success", "message": bot_response_string}

    except redis.RedisError as r_err:
        logging.error(f"Redis error for {gmail}: {str(r_err)}", exc_info=True)
        return {"type": "error", "message": f"Error connecting to session data: {str(r_err)}"}
    except Exception as e:
        logging.error(f"General error in get_reply for {gmail}: {str(e)}", exc_info=True)
        # Kembalikan dictionary yang menandakan error umum
        return {"type": "error", "message": f"An unexpected error occurred while processing your request."} # Pesan lebih umum ke user