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

import time
@st.cache_resource()
def ee_authenticate(token_name=st.secrets["refresh_token"]):
    geemap.ee_initialize(token_name=token_name, project='ee-andreanascetti', opt_url='https://earthengine-highvolume.googleapis.com')
    #ee.Authenticate()
    #ee.Initialize(project='ee-andreanascetti', opt_url='https://earthengine-highvolume.googleapis.com')


REF_YEARS = [2017, 2018, 2019, 2020, 2021, 2022]

ee_authenticate()

image = ee.Image("USGS/SRTMGL1_003")

vis_params = {
            "min": 0,
            "max": 6000,
            "palette": "terrain",
        }

st.title("Global Building Footprints")

col1, col2 = st.columns([8, 2])

st.session_state["button_state"] = True


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


with col1:
    # call to render Folium map in Streamlit
    st_data = st_folium(m, width=1000, height=1000)

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

                    CRS = processing.getUtmCRS()

                    T1 = ee.Date("2020-11-01")
                    T2 = ee.Date("2021-12-01")

                    s1 = processing.generateSentinel1Data(T1, T2)
                    s2 = processing.generateSentinel2Data(T1, T2)

                    s1clip = s1.clip(ROI)
                    s2clip = s2.clip(ROI)

                    s2_ds = xr.open_dataset(
                        ee.ImageCollection(s2clip),
                        engine='ee',
                        crs=CRS,
                        scale=10,
                        geometry=ROI
                    )

                    s1_ds = xr.open_dataset(
                        ee.ImageCollection(s1clip),
                        engine='ee',
                        crs=CRS,
                        scale=10,
                        geometry=ROI
                    )

                    s1_bands = s1_ds.to_array().to_numpy().squeeze()

                    s2_bands = s2_ds.to_array().to_numpy().squeeze()

                    print(s1_bands.shape)
                    print(s2_bands.shape)

                    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

                    model = 'weights/fusionda_10m_checkpoint15.pt'

                    net = ue.load_checkpoint(model, device)

                    s1 = np.nan_to_num(s1_bands.transpose((1, 2, 0))).astype(np.float32)

                    s2 = np.nan_to_num(s2_bands.transpose((1, 2, 0))).astype(np.float32)

                    tile_size = 256
                    dataset = ue.SceneInferenceDataset(s1, s2, tile_size)
                    pred = dataset.get_arr()

                    # @title
                    from tqdm.notebook import tqdm

                    net.eval()

                    for index in tqdm(range(len(dataset))):
                        tile = dataset.__getitem__(index)
                        x_s1, x_s2 = tile['x_s1'], tile['x_s2']
                        i, j = tile['i'], tile['j']

                        with torch.no_grad():
                            logits = net(x_s1.unsqueeze(0).to(device), x_s2.unsqueeze(0).to(device))

                        y_pred = torch.sigmoid(logits).detach()
                        y_pred = y_pred.squeeze().cpu().squeeze().numpy()
                        y_pred = y_pred[tile_size:2 * tile_size, tile_size: 2 * tile_size]

                        y_pred = np.clip(y_pred * 100, 0, 100).astype(np.uint8)
                        pred[i:i + tile_size, j:j + tile_size] = y_pred

                    # @title


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
