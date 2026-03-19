This project is an Edge-Based Predictive Maintenance System for CNC Machines that monitors machine health in real time using multiple sensors such as vibration, temperature, pressure, and sound.

The system uses threshold-based anomaly detection to identify abnormal conditions. When sensor values exceed predefined limits, the system calculates a health score (0–100) and classifies the machine status as Healthy, Warning, or Critical. This helps in quickly understanding the overall condition of the machine.

A real-time dashboard built with Streamlit displays live sensor readings, system status, maintenance suggestions, and performance trends. The system follows a cycle-based monitoring approach, allowing continuous tracking and better visualization of machine behavior over time.

Maintenance recommendations are also provided based on detected issues. For example, high vibration may indicate alignment problems, while abnormal sound levels can suggest tool wear or loose components.

The system is designed using an edge computing approach, where all processing is done locally for faster response and minimal dependency on external systems.

Overall, the project aims to reduce downtime, improve maintenance efficiency, and serve as a foundation for future upgrades like IoT integration and machine learning-based prediction.
