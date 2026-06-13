import os
import joblib
import pandas as pd
import requests
from datetime import datetime, timedelta
from fastapi import FastAPI
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==========================================
# 1. DATABASE SETUP (SQLAlchemy)
# ==========================================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local_orders.db") 

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow) # Added for SLA tracking
    store_location = Column(String, index=True)            # Added for filtering
    lens_type = Column(String, index=True)
    lens_index = Column(Float)
    sph_power = Column(Float)
    coating = Column(String)
    in_house_stock = Column(Boolean)
    status = Column(String, default="Order Placed")        # Lifecycle tracking
    sla_hours = Column(Integer)                            # SLA limit
    predicted_breach = Column(Boolean, default=False)
    delay_reason = Column(String, nullable=True)           # Delay logging

Base.metadata.create_all(bind=engine)

# ==========================================
# 2. CORE LOGIC (Inventory & SLA)
# ==========================================
def check_in_house_stock(sph_power: float, coating: str) -> bool:
    """
    Automated Inventory Check (Module 1):
    Based on historical data rules, standard powers (-2.00 to +2.00) 
    with no special coatings are kept in-house.
    """
    is_common_power = -2.00 <= sph_power <= 2.00
    is_no_coating = coating == 'None'
    return is_common_power and is_no_coating

def calculate_sla_hours(lens_type: str, in_house: bool) -> int:
    """Assigns base SLA time depending on complexity."""
    if in_house:
        return 24 # In-house delivered ASAP (1 day)
    return 72 if lens_type == 'Single Vision' else 120 # 3 days vs 5 days

def get_time_remaining(created_at: datetime, sla_hours: int, status: str) -> str:
    """Calculates dynamic countdown against SLA (Module 2)."""
    if status == "Delivered":
        return "✅ Completed"
    
    deadline = created_at + timedelta(hours=sla_hours)
    remaining = deadline - datetime.utcnow()
    hours_left = int(remaining.total_seconds() / 3600)
    
    if hours_left < 0:
        return f"🚨 Breached ({abs(hours_left)}h ago)"
    return f"⏳ {hours_left}h left"

# ==========================================
# 3. LOAD AI MODEL (Module 3)
# ==========================================
try:
    model = joblib.load('sla_predictor_model.joblib')
    encoders = joblib.load('label_encoders.joblib')
except Exception as e:
    print("Warning: Model files not found. Run train.py first.")
    model, encoders = None, None

def predict_breach(lens_type, lens_index, sph_power, coating, in_house_stock):
    if not model:
        return False
    
    input_data = pd.DataFrame([{
        'lens_type': lens_type,
        'lens_index': float(lens_index),
        'sph_power': float(sph_power),
        'coating': coating,
        'in_house_stock': bool(in_house_stock)
    }])
    
    for col in ['lens_type', 'coating']:
        input_data[col] = encoders[col].transform(input_data[col])
        
    prediction = model.predict(input_data)
    return bool(prediction[0])

# ==========================================
# 4. NOTIFICATION LOGIC
# ==========================================
def send_breach_alert(order_id):
    print(f"⚠️ ALERT TRIGGERED for Order #{order_id}...")
    
    # Twilio (WhatsApp)
    twilio_sid = os.getenv('TWILIO_ACCOUNT_SID')
    twilio_token = os.getenv('TWILIO_AUTH_TOKEN')
    if twilio_sid and twilio_token:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_sid}/Messages.json"
        payload = {
            "From": os.getenv('TWILIO_FROM'),
            "To": os.getenv('TWILIO_TO'),
            "Body": f"🚨 Eluno AI Alert: Order #{order_id} is at high risk of breaching its SLA."
        }
        requests.post(url, data=payload, auth=(twilio_sid, twilio_token))

    # Resend (Email)
    resend_key = os.getenv('RESEND_API_KEY')
    if resend_key:
        headers = {"Authorization": f"Bearer {resend_key}", "Content-Type": "application/json"}
        payload = {
            "from": "onboarding@resend.dev", 
            "to": os.getenv('ALERT_EMAIL_TO'),
            "subject": f"🚨 SLA Breach Warning: Order #{order_id}",
            "html": f"<p>AI has predicted <b>Order #{order_id}</b> will breach its SLA.</p>"
        }
        requests.post("https://api.resend.com/emails", json=payload, headers=headers)

# ==========================================
# 5. FASTAPI APPLICATION
# ==========================================
app = FastAPI(title="Eluno AI Order Management")

# ==========================================
# 6. GRADIO DASHBOARD (Frontend)
# ==========================================
def process_ui_order(location, lens_type, lens_index, sph_power, coating):
    db = SessionLocal()
    
    # 1. Automated Inventory Check
    in_house = check_in_house_stock(float(sph_power), coating)
    
    # 2. AI Breach Prediction
    will_breach = predict_breach(lens_type, lens_index, sph_power, coating, in_house)
    
    # 3. Save to DB
    new_order = Order(
        store_location=location,
        lens_type=lens_type,
        lens_index=float(lens_index),
        sph_power=float(sph_power),
        coating=coating,
        in_house_stock=in_house,
        sla_hours=calculate_sla_hours(lens_type, in_house),
        predicted_breach=will_breach
    )
    db.add(new_order)
    db.commit()
    db.refresh(new_order)
    order_id = new_order.id
    db.close()
    
    # 4. Notifications
    if will_breach:
        send_breach_alert(order_id)
        
    stock_msg = "✅ Assigned from In-House Stock." if in_house else "🏭 Sent to Manufacturing."
    risk_msg = "⚠️ HIGH RISK: AI Alert Sent." if will_breach else "✅ On Track."
    return f"Order #{order_id} Created.\nInventory: {stock_msg}\nSLA Status: {risk_msg}"

def fetch_ui_orders(status_filter, type_filter, loc_filter):
    db = SessionLocal()
    query = db.query(Order)
    
    # Apply Filters
    if status_filter != "All": query = query.filter(Order.status == status_filter)
    if type_filter != "All": query = query.filter(Order.lens_type == type_filter)
    if loc_filter != "All": query = query.filter(Order.store_location == loc_filter)
    
    orders = query.all()
    db.close()
    
    # Format for UI Table
    data = []
    for o in orders:
        sla_status = get_time_remaining(o.created_at, o.sla_hours, o.status)
        ai_risk = "⚠️ Yes" if o.predicted_breach else "No"
        in_house_str = "Yes" if o.in_house_stock else "No"
        data.append([
            o.id, o.store_location, o.lens_type, o.sph_power, in_house_str, 
            o.status, sla_status, ai_risk, o.delay_reason or "N/A"
        ])
    return data

def update_order_status(order_id, new_status, delay_reason):
    try:
        order_id = int(order_id)
    except:
        return "❌ Invalid Order ID."
        
    db = SessionLocal()
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        db.close()
        return f"❌ Order #{order_id} not found."
        
    order.status = new_status
    if delay_reason:
        order.delay_reason = delay_reason
        
    db.commit()
    db.close()
    return f"✅ Order #{order_id} updated to '{new_status}'."

# Ultimate CSS Nuke for Footer
custom_css = """
footer {display: none !important; opacity: 0 !important; height: 0 !important; margin: 0 !important; padding: 0 !important; overflow: hidden !important;}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as dashboard:
    gr.Markdown("# 👓 Eluno AI - Full Lifecycle Order Management System")
    
    with gr.Tabs():
        # --- TAB 1: ORDER INTAKE ---
        with gr.Tab("1. Order Intake"):
            gr.Markdown("### Create New Order")
            with gr.Row():
                with gr.Column():
                    ui_loc = gr.Dropdown(['Online', 'Store A (NY)', 'Store B (LA)'], value='Online', label="Store Location")
                    ui_lens = gr.Dropdown(['Single Vision', 'Progressive'], value='Single Vision', label="Lens Type")
                    ui_index = gr.Dropdown(['1.50', '1.60', '1.67', '1.74'], value='1.50', label="Lens Index")
                with gr.Column():
                    ui_sph = gr.Number(value=-1.00, label="Spherical Power")
                    ui_coating = gr.Dropdown(['None', 'Anti-Reflective', 'Blue Light', 'Blue Light + Anti-Reflective'], value='None', label="Coating")
                    
            submit_btn = gr.Button("Submit Order & Run AI Check", variant="primary")
            output_msg = gr.Textbox(label="System Output", lines=3)
            
            submit_btn.click(fn=process_ui_order, inputs=[ui_loc, ui_lens, ui_index, ui_sph, ui_coating], outputs=output_msg)

        # --- TAB 2: DASHBOARD & TRACKING ---
        with gr.Tab("2. Dashboard & SLA Tracking"):
            gr.Markdown("### Filter Active Orders")
            with gr.Row():
                filter_status = gr.Dropdown(["All", "Order Placed", "QC Failed", "Manufacturing", "Shipped", "Delivered"], value="All", label="Status")
                filter_type = gr.Dropdown(["All", "Single Vision", "Progressive"], value="All", label="Lens Type")
                filter_loc = gr.Dropdown(["All", "Online", "Store A (NY)", "Store B (LA)"], value="All", label="Location")
                refresh_btn = gr.Button("Apply Filters / Refresh", variant="primary")
                
            order_table = gr.Dataframe(
                headers=["ID", "Location", "Type", "SPH", "In-House", "Status", "SLA Timer", "AI Risk", "Delay Reason"],
                datatype=["number", "str", "str", "number", "str", "str", "str", "str", "str"],
                interactive=False
            )
            refresh_btn.click(fn=fetch_ui_orders, inputs=[filter_status, filter_type, filter_loc], outputs=order_table)

        # --- TAB 3: LIFECYCLE MANAGEMENT ---
        with gr.Tab("3. Manage Fulfillments"):
            gr.Markdown("### Update Order Status & Log Delays")
            with gr.Row():
                with gr.Column():
                    up_id = gr.Textbox(label="Order ID to Update")
                    up_status = gr.Dropdown(["Order Placed", "Manufacturing", "QC Failed", "Shipped", "Delivered"], label="New Status")
                with gr.Column():
                    up_reason = gr.Textbox(label="Reason for Delay (If applicable)", placeholder="e.g., Coating machine broken...")
                    
            update_btn = gr.Button("Update Lifecycle Status", variant="primary")
            update_msg = gr.Textbox(label="Update Result")
            
            update_btn.click(fn=update_order_status, inputs=[up_id, up_status, up_reason], outputs=update_msg)

# Mount Gradio
app = gr.mount_gradio_app(app, dashboard, path="/")