import folium
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium
import ee
import geemap
import torch
from models import urban_extractor as ue
import dask.distributed
import ee
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import rasterio as rio
import time
@st.cache_resource()
def ee_authenticate(token_name=st.secrets["refresh_token"]):
    geemap.ee_initialize(token_name=token_name, project='ee-andreanascetti', opt_url='https://earthengine-highvolume.googleapis.com')
    #ee.Authenticate()
    #ee.Initialize(project='ee-andreanascetti', opt_url='https://earthengine-highvolume.googleapis.com')


REF_YEARS = [2017, 2018, 2019, 2020, 2021, 2022]

ee_authenticate()

st.title("Global Building Footprints")

col1, col2 = st.columns([8, 2])

st.session_state["button_state"] = True


def generateMap():
    # center on Liberty Bell, add marker
    m = folium.Map(location=[39.949610, -75.150282], zoom_start=12)

    folium.TileLayer('http://mt0.google.com/vt/lyrs=y&hl=en&x={x}&y={y}&z={z}',
                     Name="Google Hybrid", detect_retina=True,  attr='GoogleMaps Hybrid').add_to(m)
    folium.TileLayer('http://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}',
                     Name="Google Satellite",detect_retina=True, attr='GoogleMaps Satellite').add_to(m)
    folium.LayerControl(position="topright").add_to(m)

    Draw(
        export=True,
        position="topleft",
        draw_options={
            "polyline": False,
            "poly": False,
            "circle": False,
            "polygon": False,
            "marker": False,
            "circlemarker": False,
            "rectangle": True,
        },
    ).add_to(m)

    #folium.raster_layers.ImageOverlay()

    return m

#TO DO check bidirectional map

# src = rio.open("/Users/andreanascetti/PycharmProjects/geoApps/planet_scope.tif")
# array = src.read()
# bounds = src.bounds
#
# x1, y1, x2, y2 = src.bounds
# bbox = [(bounds.bottom, bounds.left), (bounds.top, bounds.right)]
#
# img = folium.raster_layers.ImageOverlay(
#     name="Sentinel 1",
#     image=np.moveaxis(array, 0, -1),
#     bounds=bbox,
#     opacity=0.9,
#     interactive=True,
#     cross_origin=False,
#     zindex=1,
# )


Map = geemap.Map()
file = "/Users/andreanascetti/PycharmProjects/geoApps/landsat.tif"

#URL = "gs://gcp-public-data-landsat/LC08/01/044/034/LC08_L1TP_044034_20131228_20170307_01_T1/LC08_L1TP_044034_20131228_20170307_01_T1_B5.TIF"

#image = geemap.load_GeoTIFF(URL)

#Map.addLayer(image, {}, "Cloud Image")


with col1:
    # call to render Folium map in Streamlit
    m = generateMap()

    st_data = st_folium(m, width=1000, height=1000)

    #m2 = Map.to_streamlit(height=1000, bidirectional=False)

with col2:
    year = st.selectbox("Select a year", REF_YEARS,
                        index=None,
                        placeholder="Select reference year..."
                        )

    if year is not None:
        st.session_state["button_state"] = False

    start_processing = st.button('Processing S1 and S2 data', disabled=st.session_state["button_state"])

    if start_processing:
        st.write("Start processing")
        if st_data:
            if st_data["all_drawings"] is not None:
                draw = st_data["all_drawings"][0]
                ROI = ee.Geometry(draw['geometry'])
                with st.spinner('Wait for it...'):
                    processing = ue.appcore(ROI)
                st.success('Done!')
        else:
            st.write("No ROI selected")




@st.cache_data()
def printnew(fts):
    for draw in fts:
        print(draw)
        ROI = ee.Geometry(draw['geometry'])
        print(ROI.getInfo())

if st_data:
    if st_data["all_drawings"] is not None:
        printnew(st_data["all_drawings"])
