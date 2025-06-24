import streamlit as st
import asyncio
import websockets

st.set_page_config(page_title="Kedi MAMA Kontrol Paneli", layout="centered")

ws_url = "ws://192.168.94.144:81"  # ESP32 IP adresini buraya yaz

async def send_message(message):
    try:
        async with websockets.connect(ws_url) as websocket:
            await websocket.send(message)
    except Exception as e:
        st.error(f"WebSocket Hatası: {e}")

def streamlit_main():
    st.title("🐾 Kedi MAMA Otomasyonu Kontrol Paneli")
    st.markdown("📡 **ESP32 ile bağlantılı kontrol paneli**")

    mod = st.radio("Mod Seçimi", ["Otomatik", "Manuel"], horizontal=True)

    if mod == "Otomatik":
        if st.button("🔁 Otomatik Moda Geç"):
            asyncio.run(send_message("otomatik"))
            st.success("Otomatik mod aktif.")
    else:
        if st.button("🛠 Manuel Moda Geç"):
            asyncio.run(send_message("manuel"))
            st.success("Manuel mod aktif.")

        st.markdown("---")
        st.subheader("Manuel Kontrol")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🐟 Mama Ver (Servo)"):
                asyncio.run(send_message("manuel_servo"))
                st.info("Mama verildi (servo).")

        with col2:
            if st.button("💧 Su Ver (Pump)"):
                asyncio.run(send_message("manuel_su"))
                st.info("Su verildi (pump).")

if __name__ == "__main__":
    streamlit_main()
