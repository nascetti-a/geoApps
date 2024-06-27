import ee
import time
import geemap.foliumap as geemap
import geopandas as gpd
import streamlit as st
import folium
import streamlit as st
from folium.plugins import Draw

from streamlit_folium import st_folium

from models import urban_extractor as ue

st.set_page_config(layout="wide")

from folium.plugins import Draw

from streamlit_folium import st_folium

@st.cache(persist=True)
def ee_authenticate(token_name=st.secrets["refresh_token"]):
    geemap.ee_initialize(token_name=token_name)


st.sidebar.info(
    """
    - Web App URL: <https://streamlit.geemap.org>
    - GitHub repository: <https://github.com/giswqs/streamlit-geospatial>
    """
)

st.sidebar.title("Contact")
st.sidebar.info(
    """
    Andrea Nascetti @KTH Geoinformatics Division 
    [GitHub](https://github.com/nascetti-a) | [LinkedIn](https://www.linkedin.com/in/qiushengwu)
    """
)

st.title("Global Building Footprints")

col1, col2 = st.columns([8, 2])

@st.cache_data
def read_data(url):
    return gpd.read_file(url)


countries = 'https://github.com/giswqs/geemap/raw/master/examples/data/countries.geojson'
states = 'https://github.com/giswqs/geemap/raw/master/examples/data/us_states.json'

countries_gdf = read_data(countries)
states_gdf = read_data(states)

country_names = countries_gdf['NAME'].values.tolist()
country_names.remove('United States of America')
country_names.append('USA')
country_names.sort()
country_names = [name.replace('.', '').replace(' ', '_')
                 for name in country_names]

state_names = states_gdf['name'].values.tolist()

basemaps = list(geemap.basemaps)

Map = geemap.Map(
            basemap="HYBRID",
            plugin_Draw=False,
            Draw_export=False,
            locate_control=False,
            plugin_LatLngPopup=False,
        )

with col2:

    basemap = st.selectbox("Select a basemap", basemaps,
                           index=basemaps.index('HYBRID'))
    Map.add_basemap(basemap)

    country = st.selectbox('Select a country', country_names,
                           index=country_names.index('USA'))

    if country == 'USA':
        state = st.selectbox('Select a state', state_names,
                             index=state_names.index('Florida'))
        layer_name = state

        try:
            fc = ee.FeatureCollection(
                f'projects/sat-io/open-datasets/MSBuildings/US/{state}')
        except:
            st.error('No data available for the selected state.')

    else:
        try:
            fc = ee.FeatureCollection(
                f'projects/sat-io/open-datasets/MSBuildings/{country}')
        except:
            st.error('No data available for the selected country.')

        layer_name = country

    color = st.color_picker('Select a color', '#FF5500')

    style = {'fillColor': '00000000', 'color': color}







    #split = st.checkbox("Split-panel map")

    #if split:
    #    left = geemap.ee_tile_layer(fc.style(**style), {}, 'Left')
    #    right = left
    #    Map.split_map(left, right)
    #else:
    #    Map.addLayer(fc.style(**style), {}, layer_name)
    #
    #Map.centerObject(fc.first(), zoom=16)

    with st.expander("Data Sources"):
        st.info(
            """
            [Microsoft Building Footprints](https://gee-community-catalog.org/projects/msbuildings/)
            """
        )


with col1:

    m = Map.to_streamlit(height=1000, bidirectional=False)

    if st.button('Get center'):
        print(Map.st_map_center(m))

    if st.button('Processing S1 and S2 data'):
        if Map.st_draw_features(m) is not None and len(Map.st_draw_features(m)) > 0:
            print("got it")
            area = Map.st_draw_features(m)[0]

            geoft = ee.Geometry(area['geometry'])
            print(geoft)

            with st.spinner('Downloading S1 and S2 data...'):
                processing = ue.appcore(geoft)
                CRS = processing.getUtmCRS()
                T1 = ee.Date("2020-11-01")
                T2 = ee.Date("2021-12-01")
                s1 = processing.generateSentinel1Data(T1, T2)
                s2 = processing.generateSentinel2Data(T1, T2)
                st.success('Done!')
    #print(Map.st_map_bounds(m))
        #

        #
        #     else:
        #         st.success('No Area selected, please draw a polygon in the map!')

    #if Map.st_last_draw(m) is not None:
    #  print("got it")


    #m = folium.Map(location=[39.949610, -75.150282], zoom_start=5)
    #Draw(export=True).add_to(m)

    #c1, c2 = st.columns(2)

    #output = st_folium(Map, width=1000, height=1000)


    #st.write(output)
