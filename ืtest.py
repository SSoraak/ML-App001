import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- ฟังก์ชันสำหรับแต่ละหน้า ---

def home():
    st.title("ยินดีต้อนรับสู่แอปพลิเคชันวิเคราะห์ข้อมูล")
    st.write("ใช้แถบด้านข้างเพื่อนำทางไปยังส่วนต่างๆ ของแอปพลิเคชัน")

def data_import():
    st.title("นำเข้าข้อมูล")
    uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV หรือ Excel", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state['data'] = df
            st.success("นำเข้าข้อมูลสำเร็จ!")
            st.subheader("ตัวอย่างข้อมูล")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")

def data_display_graph():
    st.title("แสดงข้อมูลและกราฟ")
    if 'data' in st.session_state:
        df = st.session_state['data']
        st.subheader("ข้อมูลทั้งหมด")
        st.dataframe(df)

        st.subheader("สร้างกราฟ")
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numerical_cols:
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("เลือกคอลัมน์สำหรับแกน X", numerical_cols)
            with col2:
                plot_type = st.selectbox("เลือกประเภทกราฟ", ["Histogram", "Scatter Plot"])

            if plot_type == "Histogram":
                plt.figure(figsize=(10, 6))
                sns.histplot(df[x_axis], kde=True)
                st.pyplot(plt)
            elif plot_type == "Scatter Plot":
                if len(numerical_cols) > 1:
                    y_axis = st.selectbox("เลือกคอลัมน์สำหรับแกน Y", [col for col in numerical_cols if col != x_axis])
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(x=x_axis, y=y_axis, data=df)
                    st.pyplot(plt)
                else:
                    st.warning("ไม่พบคอลัมน์ตัวเลขที่เพียงพอสำหรับ Scatter Plot")
        else:
            st.warning("ไม่มีคอลัมน์ตัวเลขในข้อมูล")
    else:
        st.info("โปรดนำเข้าข้อมูลก่อน")

def dashboard():
    st.title("แดชบอร์ด")
    if 'data' in st.session_state:
        df = st.session_state['data']
        st.subheader("สถิติเบื้องต้น")
        st.write(df.describe())

        # สามารถเพิ่มการแสดงผลกราฟหรือข้อมูลสรุปอื่นๆ ได้ที่นี่
        st.subheader("ตัวอย่างการแสดงผลกราฟในแดชบอร์ด")
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numerical_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                plt.figure(figsize=(8, 4))
                sns.histplot(df[numerical_cols[0]], kde=True)
                st.pyplot(plt)
            with col2:
                plt.figure(figsize=(8, 4))
                sns.scatterplot(x=numerical_cols[0], y=numerical_cols[1], data=df)
                st.pyplot(plt)
        elif numerical_cols:
            plt.figure(figsize=(8, 4))
            sns.histplot(df[numerical_cols[0]], kde=True)
            st.pyplot(plt)
        else:
            st.warning("ไม่มีข้อมูลตัวเลขสำหรับแสดงกราฟในแดชบอร์ด")

    else:
        st.info("โปรดนำเข้าข้อมูลก่อน")

def model_building():
    st.title("สร้างโมเดล")
    if 'data' in st.session_state:
        df = st.session_state['data']
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numerical_cols) < 2:
            st.warning("ข้อมูลต้องมีคอลัมน์ตัวเลขอย่างน้อย 2 คอลัมน์เพื่อสร้างโมเดล")
            return

        target_column = st.selectbox("เลือกคอลัมน์เป้าหมาย (Target)", numerical_cols)
        feature_columns = st.multiselect("เลือกคอลัมน์คุณสมบัติ (Features)", [col for col in numerical_cols if col != target_column])

        if feature_columns and target_column:
            X = df[feature_columns]
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model_type = st.selectbox("เลือกประเภทโมเดล", ["Linear Regression"]) # เพิ่มโมเดลอื่นๆ ได้ในอนาคต

            if model_type == "Linear Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                st.session_state['model'] = model
                st.session_state['feature_cols'] = feature_columns
                st.session_state['target_col'] = target_column
                st.success("สร้างโมเดล Linear Regression สำเร็จ!")

                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                st.write(f"Mean Squared Error บนชุดข้อมูลทดสอบ: {mse:.2f}")
        else:
            st.warning("โปรดเลือกทั้งคอลัมน์เป้าหมายและคอลัมน์คุณสมบัติ")
    else:
        st.info("โปรดนำเข้าข้อมูลก่อน")

def prediction():
    st.title("ทำนายค่า")
    if 'model' in st.session_state and 'feature_cols' in st.session_state and 'target_col' in st.session_state:
        model = st.session_state['model']
        feature_cols = st.session_state['feature_cols']
        target_col = st.session_state['target_col']

        st.subheader("ป้อนค่าคุณสมบัติเพื่อทำนาย")
        input_data = {}
        for col in feature_cols:
            input_data[col] = st.number_input(f"ค่าของ {col}", value=0.0)

        if st.button("ทำนาย"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            st.subheader("ผลการทำนาย")
            st.write(f"ค่าที่ทำนายสำหรับ {target_col}: {prediction:.2f}")
    else:
        st.info("โปรดนำเข้าข้อมูลและสร้างโมเดลก่อน")

# --- แถบนำทาง ---
st.sidebar.title("แถบนำทาง")
page = st.sidebar.radio("เลือกหน้า", ["หน้าหลัก", "นำเข้าข้อมูล", "แสดงข้อมูลและกราฟ", "แดชบอร์ด", "สร้างโมเดล", "ทำนายค่า"])

# --- แสดงเนื้อหาตามหน้าที่เลือก ---
if page == "หน้าหลัก":
    home()
elif page == "นำเข้าข้อมูล":
    data_import()
elif page == "แสดงข้อมูลและกราฟ":
    data_display_graph()
elif page == "แดชบอร์ด":
    dashboard()
elif page == "สร้างโมเดล":
    model_building()
elif page == "ทำนายค่า":
    prediction()