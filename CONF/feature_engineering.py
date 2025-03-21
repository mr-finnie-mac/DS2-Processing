import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from geopy.distance import geodesic
from matplotlib.markers import MarkerStyle

title_s = 18
label_s = 16
line_w = 6

def compute_distance_to_tower(data, tower_location):
    """Compute Euclidean distance in meters to the tower."""
    distances = []
    for _, row in data.iterrows():
        point = (row["gps.lat"], row["gps.lon"])
        distance = geodesic(point, tower_location[:2]).meters  # Convert to meters
        distances.append(distance)
    return pd.Series(distances)




def compute_tower_direction(data, tower_location):
    """
    Computes azimuth and elevation angle from each point to the tower.

    Args:
        data (pd.DataFrame): DataFrame containing GPS lat, lon, and altitudeAMSL.
        tower_location (tuple): (lat, lon, altitudeAMSL) of the tower.

    Returns:
        pd.Series, pd.Series: Azimuth (degrees), Elevation angle (degrees).
    """
    tower_lat, tower_lon, tower_alt = tower_location
    azimuths, elevations = [], []

    for _, row in data.iterrows():
        # Compute azimuth (bearing)
        delta_lon = np.radians(tower_lon - row["gps.lon"])
        lat1, lat2 = np.radians(row["gps.lat"]), np.radians(tower_lat)

        x = np.sin(delta_lon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
        azimuth = np.degrees(np.arctan2(x, y)) % 360  # Normalize to 0-360 degrees
        azimuths.append(azimuth)

        # Compute elevation angle
        ground_distance = geodesic((row["gps.lat"], row["gps.lon"]), (tower_lat, tower_lon)).meters
        altitude_diff = tower_alt - row["altitudeAMSL"]
        elevation_angle = np.degrees(np.arctan2(altitude_diff, ground_distance))
        elevations.append(elevation_angle)


    return pd.Series(azimuths, index=data.index), pd.Series(elevations, index=data.index)


def compute_sinr_weighted_rssi(data):
    """Compute RSSI weighted by SINR with proper scaling."""
    sinr_scaled = np.clip(data["sinr"], -10, 30)  # Ensure values are within a reasonable range
    weighted_rssi = data["rssi"] * (1 + sinr_scaled / 30)  # Normalize effect
    return weighted_rssi

from datetime import datetime

def plot_flightpath_combined(data, tower_location):
    """
    Plots the flight path with distance-based color coding and azimuth-based directional markers.
    Saves the figure as a PDF with a timestamped filename.

    Args:
        data (pd.DataFrame): DataFrame containing lat, lon, computed distances, and azimuths.
        tower_location (tuple): (lat, lon, altitudeAMSL) of the tower.
    """
    # Ensure figures use a serif font similar to Times New Roman
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 16,
        "axes.titlesize": 20,
        "legend.fontsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15
    })
    fig, ax = plt.subplots(figsize=(8, 8))  # Square figure for consistency

    # Compute features
    distances = compute_distance_to_tower(data, tower_location)
    azimuths, _ = compute_tower_direction(data, tower_location)
    max_distance = distances.max()

    # --- Scatter plot of flight path with distance-based color ---
    sc = ax.scatter(
        data["gps.lon"], data["gps.lat"], c=distances, cmap="Reds", edgecolors="black",
        alpha=0.8, marker="o", s=80  # Print-friendly with black edges
    )
    cbar = plt.colorbar(sc, label="Distance to Tower (m)")
    cbar.ax.tick_params(labelsize=12)  # Match font sizes

    # --- Overlay rotated markers for azimuth ---
    for i in range(0, len(data)):  # Sample every few points
        lon, lat, azimuth = data.iloc[i]["gps.lon"], data.iloc[i]["gps.lat"], azimuths.iloc[i]

        # Create a rotated "line" marker
        m = MarkerStyle("|")
        m._transform.rotate_deg(-azimuth)  

        # Plot the marker
        ax.scatter(lon, lat, marker=m, color="black", alpha=0.9, s=200)

    # --- Plot tower location ---
    ax.scatter(tower_location[1], tower_location[0], color="red", marker="^", s=300, label="Tower")

    # --- Add constraint measurement (Max distance * 2) ---
    x_min, x_max = data["gps.lon"].min(), data["gps.lon"].max()
    ax.plot([x_min, x_max], [data["gps.lat"].max() + 0.0005, data["gps.lat"].max() + 0.0005], 
            linestyle="dashed", color="black", linewidth=1.5, label=f"Max Dist: {max_distance*2:.1f}m")

    # Labels and legend
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Flight Path with Distance & Direction Features")
    ax.legend(loc="upper right")  # Placed inside for better printing
    
    plt.grid(True, linestyle="dotted")

    # --- Save figure as PDF ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]  # Timestamp in milliseconds
    filename = f"features_flightpath_{timestamp}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Figure saved as {filename}")

    plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from geopy.distance import geodesic

def plot_flightpath_with_distances(data, tower_location):
    """
    Plots the flight path with distances to the tower labeled.

    Args:
        data (pd.DataFrame): DataFrame containing lat, lon, and computed distances.
        tower_location (tuple): (lat, lon, altitudeAMSL) of the tower.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute distances
    distances = compute_distance_to_tower(data, tower_location)
    
    # Scatter plot of flight path
    sc = ax.scatter(data["gps.lon"], data["gps.lat"], c=distances, cmap="coolwarm", edgecolors="black", alpha=0.8)
    plt.colorbar(sc, label="Distance to Tower (m)")

    # Add labels for distances
    for i in range(0, len(data), max(1, len(data) // 15)):  # Show every few points
        ax.text(data.iloc[i]["gps.lon"], data.iloc[i]["gps.lat"], f"{distances.iloc[i]:.1f}m",
                fontsize=13, ha="right", color="green")

    # Plot tower location
    ax.scatter(tower_location[1], tower_location[0], color="red", marker="1", s=550, label="Tower")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Flight Path with Distance to Tower")
    ax.legend()
    plt.show()



def plot_flightpath_with_diamonds(data, tower_location):
    """
    Plots the flight path with diamond markers rotated according to azimuth angles.
    Also prints azimuth values for debugging.

    Args:
        data (pd.DataFrame): DataFrame containing lat, lon, and computed azimuths.
        tower_location (tuple): (lat, lon, altitudeAMSL) of the tower.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })
    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute azimuths
    azimuths, _ = compute_tower_direction(data, tower_location)

    # Scatter plot of flight path
    ax.scatter(data["gps.lon"], data["gps.lat"], color="blue", alpha=0.6, label="Flight Path")

    # Add rotated diamond markers at regular intervals
    for i in range(0, len(data)):  # Show every few points
        lon, lat, azimuth = data.iloc[i]["gps.lon"], data.iloc[i]["gps.lat"], azimuths.iloc[i]

        # Print debug information
        print(f"Point {i}: Lon={lon}, Lat={lat}, Azimuth={azimuth:.2f}°")

        # Create a rotated diamond marker
        m = MarkerStyle("|")
        m._transform.rotate_deg(-azimuth)  # Invert rotation if needed

        # Plot the marker
        ax.scatter(lon, lat, marker=m, color="black", alpha=0.8, s=250)

    # Plot tower location
    ax.scatter(tower_location[1], tower_location[0], color="red", marker="1", s=350, label="Tower")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Flight Path with Tower Bearing (Azimuth) Feature")
    ax.legend()
    plt.show()


def plot_flightpath_with_arrows(data, tower_location):
    """
    Plots the flight path with arrows showing direction to the tower.

    Args:
        data (pd.DataFrame): DataFrame containing lat, lon, and computed azimuths.
        tower_location (tuple): (lat, lon, altitudeAMSL) of the tower.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute azimuths
    azimuths, _ = compute_tower_direction(data, tower_location)

    # Scatter plot of flight path
    ax.scatter(data["gps.lon"], data["gps.lat"], color="blue", alpha=0.6, label="Flight Path")

    # Add arrows indicating direction
    for i in range(0, len(data), max(1, len(data) // 15)):  # Show every few points
        lon, lat, azimuth = data.iloc[i]["gps.lon"], data.iloc[i]["gps.lat"], azimuths.iloc[i]
        
        dx = np.cos(np.radians(azimuth)) * 0.0002  # Adjust step size
        dy = np.sin(np.radians(azimuth)) * 0.0002
        
        ax.arrow(
            lon, lat, dx, dy, 
            head_width=0.00005, head_length=0.00005,  # Adjusted head size
            fc="black", ec="black", alpha=0.8  # Optional: add transparency
        )


    # Plot tower location
    ax.scatter(tower_location[1], tower_location[0], color="red", marker="^", s=100, label="Tower")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Flight Path with Tower Direction (Azimuth)")
    ax.legend()
    plt.show()



def plot_sinr_weighted_rssi(data):
    """
    Plots the flight path with SINR-weighted RSSI values color-coded.

    Args:
        data (pd.DataFrame): DataFrame containing lat, lon, and computed SINR-weighted RSSI.
    """

    # --- Global Font & Style Settings ---
    # plt.rcParams.update({
    #     "font.family": "serif",
    #     "font.serif": ["Times New Roman"],
    #     "axes.labelsize": 18,
    #     "axes.titlesize": 20,
    #     "legend.fontsize": 18,
    #     "xtick.labelsize": 16,
    #     "ytick.labelsize": 16,
    #     "lines.linewidth": 3,  # Thick lines
    #     "lines.markersize": 10  # Large markers
    # })
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 16,
        "axes.titlesize": 20,
        "legend.fontsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
         "lines.linewidth": 3,  # Thick lines
        "lines.markersize": 10  # Large markers
    })

    fig, ax = plt.subplots(figsize=(9, 9))  # **Square Figure**

    # Compute SINR-weighted RSSI
    sinr_weighted_rssi = compute_sinr_weighted_rssi(data)

    # Scatter plot with color-coded SINR-weighted RSSI
    sc = ax.scatter(
        data["gps.lon"], data["gps.lat"], 
        c=sinr_weighted_rssi, cmap="Reds",  # **Red gradient for print-friendliness**
        edgecolors="black", linewidth=1.2, alpha=0.8
    )
    
    # **Colorbar for SINR-weighted RSSI**
    cbar = plt.colorbar(sc, label="SINR-Weighted RSSI")
    cbar.ax.tick_params(labelsize=14)  # **Make colorbar text larger**
    
    ax.set_xlabel("Longitude", fontsize=18)
    ax.set_ylabel("Latitude", fontsize=18)
    ax.set_title("Flight Path with SINR-Weighted RSSI", fontsize=20)
    ax.grid(True, linestyle="dotted", linewidth=1.5)  # **Thicker Grid**

    plt.tight_layout()

    # --- Save figure as PDF ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
    filename = f"sinr_weighted_rssi_{timestamp}.pdf"
    plt.savefig(filename, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Figure saved as {filename}")

    plt.show()
