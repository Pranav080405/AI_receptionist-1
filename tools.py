import json

MOCK_CLINIC_SCHEDULE = {
    "hours": "9:00 AM to 5:00 PM, Monday through Friday",
    "services": "General Cleaning, Dental Checkup, Cavity Filling, Teeth Whitening",
    "open_slots_today": ["2:00 PM", "4:30 PM"]
}

def execute_clinic_tool(tool_call: dict) -> str:
    func_name = tool_call.get("function", {}).get("name")
    args = tool_call.get("function", {}).get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except:
            args = {}

    if func_name == "get_clinic_info":
        return json.dumps(MOCK_CLINIC_SCHEDULE)

    elif func_name == "book_appointment":
        time_slot = args.get("time_slot")
        name = args.get("name", "Caller")
        return json.dumps({
            "status": "confirmed",
            "message": f"Appointment booked for {name} at {time_slot}."
        })

    return json.dumps({"error": "Unknown function"})