import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from geopy.distance import geodesic

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
                fontsize=8, ha="right", color="black")

    # Plot tower location
    ax.scatter(tower_location[1], tower_location[0], color="red", marker="^", s=100, label="Tower")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Flight Path with Distance to Tower")
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
    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute SINR-weighted RSSI
    sinr_weighted_rssi = compute_sinr_weighted_rssi(data)

    # Scatter plot with color-coded SINR-weighted RSSI
    sc = ax.scatter(data["gps.lon"], data["gps.lat"], c=sinr_weighted_rssi, cmap="RdYlGn", edgecolors="black", alpha=0.8)
    plt.colorbar(sc, label="SINR-Weighted RSSI")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Flight Path with SINR-Weighted RSSI")
    plt.show()
