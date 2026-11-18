# HƯỚNG DẪN CẤU HÌNH FIREBASE CHO VIDEO DUBBER (1 TÀI KHOẢN = 1 MÁY)

Hệ thống quản lý tài khoản của **Video Dubber** sử dụng cơ sở dữ liệu đám mây **Firebase** (Google Cloud) hoàn toàn miễn phí.
Mỗi tài khoản người dùng chỉ có thể đăng nhập trên **1 máy tính duy nhất**. Khi người dùng cố đăng nhập trên máy tính khác, hệ thống sẽ tự động chặn và thông báo khóa máy.

---

## BƯỚC 1: Tạo dự án Firebase (Miễn phí)
1. Truy cập vào trang quản trị: [https://console.firebase.google.com/](https://console.firebase.google.com/)
2. Bấm **"Thêm dự án"** (Add project), đặt tên dự án (ví dụ: `my-video-dubber`), tắt Google Analytics rồi bấm **"Tạo dự án"**.

---

## BƯỚC 2: Bật xác thực Email / Password
1. Trong menu bên trái, vào **Build (Xây dựng)** > **Authentication**.
2. Bấm **Get started (Bắt đầu)**.
3. Trong tab **Sign-in method**, chọn **Email/Password**.
4. Bật công tắc **Enable (Bật)** ở dòng đầu tiên > Bấm **Save (Lưu)**.
5. Chuyển sang tab **Users (Người dùng)**:
   - Bấm **Add user (Thêm người dùng)** để tạo tài khoản (Email & Mật khẩu) cấp cho khách hàng / người dùng của bạn.

---

## BƯỚC 3: Tạo cơ sở dữ liệu Firestore (Lưu trữ khóa máy)
1. Trong menu bên trái, vào **Build** > **Firestore Database**.
2. Bấm **Create database (Tạo cơ sở dữ liệu)**.
3. Chọn vị trí server (ví dụ `asia-southeast1 - Singapore`) > Chọn chế độ **Start in test mode** hoặc **Production mode**.
4. Vào tab **Rules (Quy tắc)** của Firestore, dán nội dung sau rồi bấm **Publish**:
```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /users/{userId} {
      // Cho phép người dùng đã đăng nhập đọc và ghi thông tin máy của chính mình
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
  }
}
```

---

## BƯỚC 4: Lấy thông tin cấu hình và nhập vào app
1. Bấm vào biểu tượng bánh răng **Cài đặt dự án (Project settings)** ở góc trên bên trái màn hình Firebase Console.
2. Tại tab **General (Chung)**:
   - Sao chép **Web API Key** (Dạng: `AIzaSy...`).
   - Sao chép **Project ID** (Dạng: `my-video-dubber-12345`).
3. Mở ứng dụng **Video Dubber**:
   - Tại màn hình đăng nhập, bấm vào chữ **"⚙ Cấu hình Firebase"**.
   - Dán **Web API Key** và **Project ID** vào 2 ô tương ứng > Bấm **Lưu cấu hình**.
   - *(Hoặc bạn có thể tạo file `firebase_config.json` đặt ngay cạnh file `VideoDubber.exe` với nội dung:)*
   ```json
   {
     "apiKey": "AIzaSy...",
     "projectId": "my-video-dubber-12345"
   }
   ```

---

## CƠ CHẾ KHÓA 1 TÀI KHOẢN TRÊN 1 MÁY HOẠT ĐỘNG THẾ NÀO?
1. **Lần đầu đăng nhập:** Ứng dụng đọc mã phần cứng của máy tính hiện tại (SHA-256 kết hợp từ địa chỉ MAC của card mạng và Windows MachineGuid của bo mạch chủ) và lưu vào hồ sơ tài khoản trên Firestore.
2. **Đăng nhập trên đúng máy đó:** Hệ thống đối chiếu mã máy khớp 100% -> Cho phép sử dụng bình thường.
3. **Đăng nhập trên máy khác:** Hệ thống phát hiện mã máy không trùng khớp -> **Từ chối đăng nhập**, hiển thị cảnh báo:
   `"⛔ TÀI KHOẢN ĐÃ ĐƯỢC ĐĂNG KÝ TRÊN MÁY KHÁC! Mỗi tài khoản chỉ được sử dụng trên 1 máy tính."`
4. **Khi khách hàng muốn đổi máy:** Quản trị viên chỉ cần vào Firestore Database > Collection `users` > Tìm `uid` của user đó và **xóa trường `machine_id`** (hoặc đặt giá trị rỗng `""`). Khi user đăng nhập trên máy mới, máy mới sẽ tự động được gán làm máy chính chủ.
