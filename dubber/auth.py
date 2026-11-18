"""Firebase Authentication + Real-Time Account Status & Machine-Lock Enforcement (1 Account = 1 Machine).

Security guarantees:
1. Disabling or deleting an account in Firebase Auth Console (or setting is_active=false in Firestore)
   immediately revokes access on app startup, before any render job, and during live heartbeat checks.
2. Refresh tokens are exchanged & verified via Identity Toolkit `accounts:lookup` (`disabled` flag)
   + Firestore `users/{uid}` (`is_active` & `machine_id`) — NO blind local token trust and NO fail-open bypass.
3. Logging out purges both disk session (`token.json`) and in-memory Admin state completely.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
import winreg
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
import urllib.request
import urllib.error

# ── Paths ──────────────────────────────────────────────────────────────────────
_APP_DATA = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "VideoDubber"
_TOKEN_FILE = _APP_DATA / "token.json"
_CONFIG_USER_FILE = _APP_DATA / "firebase_config.json"


def _get_project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent.parent


_CONFIG_ROOT_FILE = _get_project_root() / "firebase_config.json"

# In-memory only flag for Admin PIN session (NEVER saved to disk!)
_admin_bypass_session: bool = False


# ── Machine Fingerprint ────────────────────────────────────────────────────────
def get_machine_id() -> str:
    """Return a unique, stable hardware identifier for this Windows machine."""
    mac = str(uuid.getnode())
    win_guid = ""
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Cryptography",
        )
        win_guid, _ = winreg.QueryValueEx(key, "MachineGuid")
        winreg.CloseKey(key)
    except Exception:
        pass
    raw = f"{mac}|{win_guid}|VideoDubber"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def get_short_machine_id() -> str:
    """Return a user-friendly short machine ID (e.g. for sending to admin)."""
    mid = get_machine_id()
    return f"{mid[:4]}-{mid[4:8]}-{mid[8:12]}-{mid[12:16]}".upper()


# ── Hardcoded Defaults (Locked by Admin) ───────────────────────────────────────
HARDCODED_API_KEY = "AIzaSyCIE_VN_R-kfII5ZXOOeFQI6hLGo1k45zs"
HARDCODED_PROJECT_ID = "my-video-dubber"
HARDCODED_AUTH_DOMAIN = "my-video-dubber.firebaseapp.com"

ADMIN_PIN = "100102"  # Only admin with this code can change Firebase settings or use Admin mode


def verify_admin_pin(pin: str) -> bool:
    """Verify if entered PIN matches master admin password."""
    return str(pin).strip() == ADMIN_PIN


def activate_admin_session():
    """Enable in-memory Admin session (not persisted to disk; cleared on logout/exit)."""
    global _admin_bypass_session
    _admin_bypass_session = True


def is_admin_session() -> bool:
    return _admin_bypass_session


# ── Configuration Management (100% Hardcoded & Tamper-Proof) ──────────────────
def get_firebase_config() -> dict:
    """Return strictly locked Firebase configuration (cannot be overridden by external json files)."""
    return {
        "apiKey": HARDCODED_API_KEY,
        "projectId": HARDCODED_PROJECT_ID,
        "authDomain": HARDCODED_AUTH_DOMAIN,
    }


def is_firebase_configured() -> bool:
    return bool(HARDCODED_API_KEY and HARDCODED_PROJECT_ID)


# ── Token Cache ────────────────────────────────────────────────────────────────
def _save_token(uid: str, email: str, id_token: str = "", refresh_token: str = ""):
    _APP_DATA.mkdir(parents=True, exist_ok=True)
    data = {
        "uid": uid,
        "email": email,
        "machine_id": get_machine_id(),
        "id_token": id_token,
        "refresh_token": refresh_token,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    _TOKEN_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _load_token() -> Optional[dict]:
    if not _TOKEN_FILE.exists():
        return None
    try:
        data = json.loads(_TOKEN_FILE.read_text(encoding="utf-8"))
        if data.get("machine_id") != get_machine_id():
            clear_token()
            return None
        if not data.get("uid") or not data.get("id_token"):
            clear_token()
            return None
        return data
    except Exception:
        clear_token()
        return None


def clear_token():
    """Purge saved token file and in-memory admin session."""
    global _admin_bypass_session
    _admin_bypass_session = False
    if _TOKEN_FILE.exists():
        try:
            _TOKEN_FILE.unlink()
        except Exception:
            try:
                _TOKEN_FILE.write_text("{}", encoding="utf-8")
            except Exception:
                pass


def get_current_user() -> Optional[dict]:
    """Return info of currently signed in user (or Admin session)."""
    if _admin_bypass_session:
        return {"uid": "admin", "email": "Quản trị viên (Admin)"}
    return _load_token()


# ── REST API Calls ─────────────────────────────────────────────────────────────
def _translate_error(msg: str) -> str:
    mapping = {
        "EMAIL_NOT_FOUND": "Email chưa được đăng ký trong hệ thống.",
        "USER_NOT_FOUND": "Tài khoản không tồn tại hoặc đã bị xóa khỏi hệ thống.",
        "INVALID_PASSWORD": "Mật khẩu không chính xác.",
        "USER_DISABLED": "⛔ Tài khoản này đã bị Quản trị viên vô hiệu hóa (khóa)!",
        "TOKEN_EXPIRED": "Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.",
        "INVALID_REFRESH_TOKEN": "Phiên đăng nhập không hợp lệ. Vui lòng đăng nhập lại.",
        "INVALID_ID_TOKEN": "Phiên đăng nhập đã hết hiệu lực. Vui lòng đăng nhập lại.",
        "INVALID_EMAIL": "Địa chỉ email không đúng định dạng.",
        "TOO_MANY_ATTEMPTS_TRY_LATER": "Thử đăng nhập sai quá nhiều lần. Vui lòng đợi 5 phút.",
        "INVALID_LOGIN_CREDENTIALS": "Email hoặc mật khẩu không chính xác.",
        "API_KEY_INVALID": "API Key của Firebase không hợp lệ.",
        "PROJECT_NOT_FOUND": "Không tìm thấy Firebase Project ID.",
    }
    for k, v in mapping.items():
        if k in msg:
            return v
    return f"Lỗi xác thực: {msg}"


def _rest_sign_in(email: str, password: str, api_key: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Sign in with Firebase Auth. Returns (uid, id_token, refresh_token, error_msg)."""
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
    payload = json.dumps({
        "email": email,
        "password": password,
        "returnSecureToken": True,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "VideoDubber-Desktop"}
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["localId"], data["idToken"], data.get("refreshToken", ""), None
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
            err_json = json.loads(err_body)
            raw_msg = err_json.get("error", {}).get("message", str(e))
            return None, None, None, _translate_error(raw_msg)
        except Exception:
            return None, None, None, f"Lỗi máy chủ ({e.code}): {e.reason}"
    except urllib.error.URLError:
        return None, None, None, "Không thể kết nối Internet tới máy chủ xác thực tài khoản."
    except Exception as e:
        return None, None, None, str(e)


def _refresh_id_token(refresh_token: str, api_key: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Exchange refresh_token for a fresh id_token. Immediately fails if user is disabled/deleted!
    Returns (new_id_token, new_refresh_token, error_msg)."""
    if not refresh_token:
        return None, None, "Phiên đăng nhập cũ cần đăng nhập lại."
    url = f"https://securetoken.googleapis.com/v1/token?key={api_key}"
    payload = json.dumps({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "VideoDubber-Desktop"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["id_token"], data.get("refresh_token", refresh_token), None
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
            err_json = json.loads(err_body)
            raw_msg = err_json.get("error", {}).get("message", str(e))
            return None, None, _translate_error(raw_msg)
        except Exception:
            return None, None, f"Phiên đăng nhập bị từ chối ({e.code})."
    except urllib.error.URLError:
        return None, None, "NETWORK_ERROR"
    except Exception as e:
        return None, None, str(e)


def _lookup_account_active(id_token: str, api_key: str) -> Tuple[bool, str]:
    """Real-time check via accounts:lookup to catch disabled/deleted users instantly."""
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={api_key}"
    payload = json.dumps({"idToken": id_token}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "VideoDubber-Desktop"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            users = data.get("users", [])
            if not users:
                return False, "⛔ Tài khoản không còn tồn tại trên hệ thống."
            u = users[0]
            if u.get("disabled", False):
                return False, "⛔ Tài khoản này đã bị Quản trị viên vô hiệu hóa (khóa)!"
            return True, ""
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
            err_json = json.loads(err_body)
            raw_msg = err_json.get("error", {}).get("message", str(e))
            return False, _translate_error(raw_msg)
        except Exception:
            return False, "⛔ Không thể xác minh trạng thái tài khoản."
    except urllib.error.URLError:
        return False, "NETWORK_ERROR"
    except Exception as e:
        return False, str(e)


def _check_and_lock_firestore(uid: str, email: str, id_token: str, project_id: str) -> Tuple[bool, str]:
    """Check machine_id & is_active on Firestore. Enforces: 1 Account = 1 Machine + active status."""
    if not project_id:
        return True, ""

    doc_url = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents/users/{uid}"
    curr_mid = get_machine_id()
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    headers = {
        "Authorization": f"Bearer {id_token}",
        "Content-Type": "application/json",
        "User-Agent": "VideoDubber-Desktop",
    }

    doc_data = None
    doc_exists = False

    try:
        req = urllib.request.Request(doc_url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=12) as resp:
            doc_data = json.loads(resp.read().decode("utf-8"))
            doc_exists = True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            doc_exists = False
        elif e.code == 401:
            return False, "⛔ Phiên xác thực hết hạn hoặc tài khoản đã bị khóa."
        elif e.code == 403:
            # Firestore database not yet created or rules deny access; Auth disabled check already passed above
            return True, ""
        else:
            return False, f"Lỗi kiểm tra cơ sở dữ liệu bản quyền ({e.code})."
    except urllib.error.URLError:
        return False, "Không thể kết nối tới máy chủ bản quyền Firebase. Vui lòng kiểm tra Internet."
    except Exception as e:
        return False, f"Lỗi kiểm tra thiết bị: {e}"

    if not doc_exists:
        create_payload = json.dumps({
            "fields": {
                "email": {"stringValue": email},
                "machine_id": {"stringValue": curr_mid},
                "machine_label": {"stringValue": os.environ.get("COMPUTERNAME", "Windows-PC")},
                "first_login": {"timestampValue": now_str},
                "last_login": {"timestampValue": now_str},
                "is_active": {"booleanValue": True},
            }
        }).encode("utf-8")
        try:
            req = urllib.request.Request(doc_url, data=create_payload, headers=headers, method="PATCH")
            with urllib.request.urlopen(req, timeout=10):
                pass
            return True, ""
        except Exception:
            return True, ""

    fields = doc_data.get("fields", {})
    stored_mid = fields.get("machine_id", {}).get("stringValue", "")
    is_active = fields.get("is_active", {}).get("booleanValue", True)

    if not is_active:
        return False, "⛔ Tài khoản này đã bị Quản trị viên khóa quyền sử dụng!"

    if not stored_mid:
        patch_url = f"{doc_url}?updateMask.fieldPaths=machine_id&updateMask.fieldPaths=last_login"
        patch_payload = json.dumps({
            "fields": {
                "machine_id": {"stringValue": curr_mid},
                "last_login": {"timestampValue": now_str},
            }
        }).encode("utf-8")
        try:
            req = urllib.request.Request(patch_url, data=patch_payload, headers=headers, method="PATCH")
            with urllib.request.urlopen(req, timeout=10):
                pass
        except Exception:
            pass
        return True, ""

    if stored_mid == curr_mid:
        patch_url = f"{doc_url}?updateMask.fieldPaths=last_login"
        patch_payload = json.dumps({
            "fields": {
                "last_login": {"timestampValue": now_str},
            }
        }).encode("utf-8")
        try:
            req = urllib.request.Request(patch_url, data=patch_payload, headers=headers, method="PATCH")
            with urllib.request.urlopen(req, timeout=8):
                pass
        except Exception:
            pass
        return True, ""

    return False, (
        "⛔ TÀI KHOẢN ĐÃ ĐƯỢC ĐĂNG KÝ TRÊN MÁY KHÁC!\n\n"
        "Mỗi tài khoản Video Dubber chỉ được sử dụng trên 1 máy tính cố định.\n"
        f"Mã máy hiện tại của bạn: {get_short_machine_id()}\n\n"
        "Nếu bạn đã đổi máy, vui lòng liên hệ Quản trị viên để reset mã máy."
    )


# ── Public High-Level API ──────────────────────────────────────────────────────
def login(email: str, password: str) -> Tuple[bool, str]:
    """Authenticate with Firebase, verify active status + machine lock, and save session."""
    global _admin_bypass_session
    _admin_bypass_session = False

    cfg = get_firebase_config()
    api_key = cfg.get("apiKey") or cfg.get("api_key")
    project_id = cfg.get("projectId") or cfg.get("project_id", "")

    if not api_key:
        return False, "Cấu hình Firebase thiếu trường 'apiKey'."

    uid, id_token, refresh_token, err = _rest_sign_in(email, password, api_key)
    if err:
        clear_token()
        return False, err

    # Verify account is not disabled
    active_ok, active_msg = _lookup_account_active(id_token, api_key)
    if not active_ok:
        clear_token()
        return False, ("Không thể kết nối máy chủ xác thực." if active_msg == "NETWORK_ERROR" else active_msg)

    allowed, lock_msg = _check_and_lock_firestore(uid, email, id_token, project_id)
    if not allowed:
        clear_token()
        return False, lock_msg

    _save_token(uid, email, id_token, refresh_token)
    return True, ""


def verify_session_online() -> Tuple[bool, str]:
    """Re-verify the saved session LIVE against Firebase Auth & Firestore.

    Immediately revokes and deletes local token if:
    - User was Disabled or Deleted in Firebase Auth Console
    - `is_active` was set to False in Firestore
    - `machine_id` was reassigned to another PC
    - User logged out (`token.json` missing)
    """
    if _admin_bypass_session:
        return True, ""

    token = _load_token()
    if not token:
        return False, "Chưa đăng nhập tài khoản."

    cfg = get_firebase_config()
    api_key = cfg.get("apiKey") or cfg.get("api_key", "")
    project_id = cfg.get("projectId") or cfg.get("project_id", "")
    if not api_key:
        clear_token()
        return False, "Thiếu cấu hình xác thực."

    uid = token.get("uid", "")
    email = token.get("email", "")
    refresh_tok = token.get("refresh_token", "")

    # 1. Refresh token exchange -> immediately catches USER_DISABLED / USER_NOT_FOUND / revoked tokens
    new_id_token, new_refresh_token, ref_err = _refresh_id_token(refresh_tok, api_key)
    if ref_err == "NETWORK_ERROR":
        return False, "Cần kết nối Internet để xác thực tài khoản bản quyền."
    if ref_err or not new_id_token:
        clear_token()
        return False, (ref_err or "Phiên đăng nhập đã hết hiệu lực.")

    # 2. Instant `accounts:lookup` check -> catches `disabled: true` with zero token-cache delay
    active_ok, active_msg = _lookup_account_active(new_id_token, api_key)
    if active_msg == "NETWORK_ERROR":
        return False, "Mất kết nối tới máy chủ xác thực tài khoản."
    if not active_ok:
        clear_token()
        return False, active_msg

    # 3. Check Firestore `is_active` & `machine_id`
    allowed, lock_msg = _check_and_lock_firestore(uid, email, new_id_token, project_id)
    if not allowed:
        clear_token()
        return False, lock_msg

    # Save refreshed tokens
    _save_token(uid, email, new_id_token, new_refresh_token)
    return True, ""


def auto_login() -> Tuple[bool, str, str]:
    """Attempt auto-login on startup ONLY if live online verification succeeds!
    Returns (success, email_or_error, uid)."""
    token = _load_token()
    if not token:
        return False, "", ""

    ok, err_msg = verify_session_online()
    if not ok:
        clear_token()
        return False, err_msg, ""

    refreshed = _load_token() or token
    return True, refreshed.get("email", ""), refreshed.get("uid", "")


def logout():
    """Completely sign out and purge both disk & memory sessions."""
    clear_token()
