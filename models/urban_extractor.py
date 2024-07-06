import math
import time
import ee
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from collections import OrderedDict
import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS as rasterio_CRS


class ROIProcessor:
    """
    A class to process satellite imagery data from Sentinel-1 and Sentinel-2 using a deep learning model.

    The ROIProcessor class encapsulates the workflow of defining a region of interest (ROI), fetching data from
    Sentinel-1 and Sentinel-2 satellite imagery, and processing this data using a pre-trained deep learning model
    for scene inference.

    Attributes:
        ROI (ee.Geometry): The region of interest defined by a polygon.
        processing (appcore): The application core instance for processing.
        CRS (str): The coordinate reference system for the ROI.
        T1 (ee.Date): The start date for the data fetching period.
        T2 (ee.Date): The end date for the data fetching period (one year after T1).
        s1 (ee.ImageCollection): The Sentinel-1 image collection.
        s2 (ee.ImageCollection): The Sentinel-2 image collection.
        s1_bands (numpy.ndarray): The processed bands from the Sentinel-1 data.
        s2_bands (numpy.ndarray): The processed bands from the Sentinel-2 data.
        model_path (str): The file path to the pre-trained model checkpoint.
        tile_size (int): The size of the tiles for scene inference.
        device (torch.device): The device to run the model on (CUDA, MPS, or CPU).
        net (torch.nn.Module): The loaded deep learning model for scene inference.

    Methods:
        __init__(poly, start_date="2020", model_path='weights/fusionda_10m_checkpoint15.pt', tile_size=256):
            Initializes the ROIProcessor with the given polygon, start date, model path, and tile size.

        load_model():
            Loads the pre-trained deep learning model.

        fetch_data():
            Fetches and processes Sentinel-1 and Sentinel-2 data for the defined ROI and time period.

        process_data():
            Processes the fetched data using the loaded model and performs scene inference.

        visualize_results(pred, s2, dataset):
            Visualizes the results of the scene inference along with the Sentinel-2 imagery.

        save_as_geotiff(pred, output_path):
            Saves the inference result as a GeoTIFF file.

        run():
            Orchestrates the entire processing workflow from data fetching to visualization and measures execution time.

    Usage:
        poly = ...  # Define your polygon coordinates here
        processor = ROIProcessor(poly)
        processor.run()
    """

    def __init__(self, poly, start_date="2020", model_path='weights/fusionda_10m_checkpoint15.pt', tile_size=256):
        self.ROI = ee.Geometry(poly)
        self.processing = appcore(self.ROI)
        self.CRS = self.processing.getUtmCRS()
        self.T1 = ee.Date(start_date)
        self.T2 = self.T1.advance(1, "year")
        self.s1 = None
        self.s2 = None
        self.s1_bands = None
        self.s2_bands = None
        self.model_path = model_path
        self.tile_size = tile_size
        self.device = self.select_device()
        self.net = self.load_model()

    def select_device(self):
        if torch.cuda.is_available():
            print("CUDA device found. Using GPU.")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("MPS device found. Using Apple Silicon GPU.")
            return torch.device("mps")
        else:
            print("No GPU found. Using CPU.")
            return torch.device("cpu")

    def load_model(self):
        return load_checkpoint(self.model_path, self.device)

    def fetch_data(self):
        self.s1 = self.processing.generateSentinel1Data(self.T1, self.T2)
        self.s2 = self.processing.generateSentinel2Data(self.T1, self.T2)
        s1clip = self.s1.clip(self.ROI)
        s2clip = self.s2.clip(self.ROI)

        s2_ds = xr.open_dataset(
            ee.ImageCollection(s2clip),
            engine='ee',
            crs=self.CRS,
            scale=10,
            geometry=self.ROI
        )

        s1_ds = xr.open_dataset(
            ee.ImageCollection(s1clip),
            engine='ee',
            crs=self.CRS,
            scale=10,
            geometry=self.ROI
        )

        #TO DO fix exporting and saving S1 and S2 data
        s1_ds.VV.rename({"X": "x", "Y": 'y'}).transpose('time', 'y', 'x').rio.to_raster("planet_scope.tif")

        self.s1_bands = s1_ds.to_array().to_numpy().squeeze()
        self.s2_bands = s2_ds.to_array().to_numpy().squeeze()

        print(self.s1_bands.shape)
        print(self.s2_bands.shape)

    def process_data(self):
        s1 = np.nan_to_num(self.s1_bands.transpose((1, 2, 0))).astype(np.float32)
        s2 = np.nan_to_num(self.s2_bands.transpose((1, 2, 0))).astype(np.float32)
        dataset = SceneInferenceDataset(s1, s2, self.tile_size)
        pred = dataset.get_arr()

        self.net.eval()
        for index in tqdm(range(len(dataset))):
            tile = dataset.__getitem__(index)
            x_s1, x_s2 = tile['x_s1'], tile['x_s2']
            i, j = tile['i'], tile['j']

            with torch.no_grad():
                logits = self.net(x_s1.unsqueeze(0).to(self.device), x_s2.unsqueeze(0).to(self.device))

            y_pred = torch.sigmoid(logits).detach()
            y_pred = y_pred.squeeze().cpu().squeeze().numpy()
            y_pred = y_pred[self.tile_size:2 * self.tile_size, self.tile_size: 2 * self.tile_size]
            y_pred = np.clip(y_pred * 100, 0, 100).astype(np.uint8)
            pred[i:i + self.tile_size, j:j + self.tile_size] = y_pred

        self.visualize_results(pred, s2, dataset)
        self.save_as_geotiff(pred.transpose(), "output.tif")

    def visualize_results(self, pred, s2, dataset):
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        fig.tight_layout()
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        s2_rgb = s2[:dataset.m, :dataset.n, [2, 1, 0]]
        axs[0].imshow(np.clip(s2_rgb / 0.4, 0, 1))
        axs[1].imshow(pred.transpose(), cmap='gray')
        fig.show()

        plt.imshow(pred)
        plt.show()

        print(pred.shape)

    # reading in geotiff file as numpy array
    def read_tif(self, file):
        with rasterio.open(file) as dataset:
            arr = dataset.read()  # (bands X height X width)
            transform = dataset.transform
            crs = dataset.crs
        return arr.transpose((1, 2, 0)), transform, crs

    def save_as_geotiff(self, pred, output_path):

        data, tt, crs = self.read_tif('planet_scope.tif')
        # Transform the ROI geometry to a bounding box
        if len(pred.shape) == 3:
            height, width, bands = pred.shape
        else:
            height, width = pred.shape
            bands = 1
            pred = pred[:, :, None]
        with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width,
                           count=bands, dtype=pred.dtype, crs=crs,
                           transform=tt,
                           ) as dst:
            for i in range(bands):
                dst.write(pred[:, :, i], i + 1)

        print(f"Inference saved as GeoTIFF: {output_path}")

    def run(self):
        start = time.time()
        self.fetch_data()
        self.process_data()
        end = time.time()
        print("Processing time:", end - start)


class appcore():
    def __init__(self, GEOMETRY):
        self.GEOMETRY = GEOMETRY
        self.POLARIZATIONS = ['VV', 'VH']
        self.SPECTRAL_BANDS = ['B2', 'B3', 'B4', 'B8']
        self.MAX_CLOUD_PROBABILITY = 80

    # This is true for lng between -180 and 180

    def getUtmCRS(self):
        coords = self.GEOMETRY.centroid(0.1).coordinates().getInfo();
        lat = coords[1]
        lng = coords[0]

        zoneNumber = (math.floor((lng + 180) / 6) % 60) + 1
        hemNumber = 7 if lat < 0 else 6
        return 'EPSG:32' + str(hemNumber) + str(zoneNumber).zfill(2)

    def generateSentinel2Data(self, fromDate, toDate):
        geom = self.GEOMETRY

        def func_heu(img):
            return img.set('geom', geom)

        s2 = ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterDate(fromDate, toDate).filterBounds(geom).map(
            func_heu)

        s2Clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
            .filterDate(fromDate, toDate) \
            .filterBounds(geom)

        joinCondition = ee.Filter.equals(**{'leftField': 'system:index', 'rightField': 'system:index'})
        s2 = ee.Join.saveFirst('cloudProbability').apply(**{
            'primary': s2,
            'secondary': s2Clouds,
            'condition': joinCondition
        })

        def func_jin(img):
            return ee.Image(img).addBands(ee.Image(img.get('cloudProbability')))

        s2 = s2.map(func_jin)

        def maskClouds(img):
            noClouds = ee.Image(img.get('cloudProbability')).lt(self.MAX_CLOUD_PROBABILITY)
            return ee.Image(img).updateMask(noClouds)

        s2noClouds = s2.map(maskClouds)
        s2Composite = ee.ImageCollection(s2noClouds) \
            .median() \
            .divide(10000) \
            .clamp(0, 1) \
            .unmask() \
            .float()

        return s2Composite.select(self.SPECTRAL_BANDS)

    def generateSentinel1Data(self, fromDate, toDate):
        geom = self.GEOMETRY
        POLARIZATIONS = ['VV', 'VH']

        s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterBounds(geom) \
            .filterDate(fromDate, toDate)

        # selecting polarizations and IW imagery
        s1 = s1.filterMetadata('instrumentMode', 'equals', 'IW').select(POLARIZATIONS)
        s1 = s1.filterMetadata('transmitterReceiverPolarisation', 'equals', POLARIZATIONS)

        # masking backscatter lower than -25 dB

        def func_kgd(img):
            img = ee.Image(img)
            notNoise = img.gte(-25)
            return img.updateMask(notNoise)

        s1 = s1.map(func_kgd)

        orbit = None
        asc = s1.filterMetadata('orbitProperties_pass', 'equals', 'ASCENDING')
        desc = s1.filterMetadata('orbitProperties_pass', 'equals', 'DESCENDING')

        # TODO: use both ASC and DSC
        s1 = ee.Algorithms.If(ee.Number(asc.size()).gt(desc.size()), asc, desc)
        s1 = ee.ImageCollection(s1)

        def func_yhj(img):
            return ee.Number(ee.Image(img).get('relativeOrbitNumber_start'))

        orbitNumbers = s1.toList(s1.size()).map(func_yhj)

        orbitNumbers = orbitNumbers.distinct().getInfo()
        # print(orbitNumbers)

        means = ee.ImageCollection([])
        for i in orbitNumbers:
            colOrbit = s1.filterMetadata('relativeOrbitNumber_start', 'equals', i)
            mean = colOrbit.mean()
            means = means.merge(ee.ImageCollection([mean]))

        meanMosaic = means.mosaic() \
            .unitScale(-25, 0) \
            .clamp(0, 1) \
            .unmask() \
            .float()

        return meanMosaic.select(self.POLARIZATIONS)


class DualStreamUNet(nn.Module):

    def __init__(self):
        super(DualStreamUNet, self).__init__()
        out = 1
        topology = [64, 128]
        out_dim = topology[0]

        # sentinel-1 sar unet stream
        sar_in = 2
        self.sar_stream = UNet(n_channels=sar_in, n_classes=out, topology=topology, enable_outc=False)
        self.sar_in = sar_in
        self.sar_out_conv = OutConv(out_dim, out)

        # sentinel-2 optical unet stream
        optical_in = 4
        self.optical_stream = UNet(n_channels=optical_in, n_classes=out, topology=topology, enable_outc=False)
        self.optical_in = optical_in
        self.optical_out_conv = OutConv(out_dim, out)

        # out block combining unet outputs
        fusion_out_dim = 2 * out_dim
        self.fusion_out_conv = OutConv(fusion_out_dim, out)

    def forward(self, x_sar: torch.Tensor, x_optical: torch.Tensor) -> torch.Tensor:

        # sar
        features_sar = self.sar_stream(x_sar)

        # optical
        features_optical = self.optical_stream(x_optical)

        features_fusion = torch.cat((features_sar, features_optical), dim=1)
        logits_fusion = self.fusion_out_conv(features_fusion)
        return logits_fusion


class UNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, topology: list,
                 enable_outc=True):
        super(UNet, self).__init__()

        first_chan = topology[0]
        self.inc = InConv(n_channels, first_chan, DoubleConv)
        self.enable_outc = enable_outc
        self.outc = OutConv(first_chan, n_classes)

        # Variable scale
        down_topo = topology
        down_dict = OrderedDict()
        n_layers = len(down_topo)
        up_topo = [first_chan]
        up_dict = OrderedDict()

        # Downward layers
        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            in_dim = down_topo[idx]
            out_dim = down_topo[idx + 1] if is_not_last_layer else down_topo[idx]
            layer = Down(in_dim, out_dim, DoubleConv)
            down_dict[f'down{idx + 1}'] = layer
            up_topo.append(out_dim)
        self.down_seq = nn.ModuleDict(down_dict)

        # Upward layers
        for idx in reversed(range(n_layers)):
            is_not_last_layer = idx != 0
            x1_idx = idx
            x2_idx = idx - 1 if is_not_last_layer else idx
            in_dim = up_topo[x1_idx] * 2
            out_dim = up_topo[x2_idx]
            layer = Up(in_dim, out_dim, DoubleConv)
            up_dict[f'up{idx + 1}'] = layer

        self.up_seq = nn.ModuleDict(up_dict)

    def forward(self, x1, x2=None):
        x = x1 if x2 is None else torch.cat((x1, x2), 1)

        x1 = self.inc(x)

        inputs = [x1]
        # Downward U:
        for layer in self.down_seq.values():
            out = layer(inputs[-1])
            inputs.append(out)

        # Upward U:
        inputs.reverse()
        x1 = inputs.pop(0)
        for idx, layer in enumerate(self.up_seq.values()):
            x2 = inputs[idx]
            x1 = layer(x1, x2)

        out = self.outc(x1) if self.enable_outc else x1

        return out


# sub-parts of the U-Net model
class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(InConv, self).__init__()
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Down, self).__init__()

        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.detach().size()[2] - x1.detach().size()[2]
        diffX = x2.detach().size()[3] - x1.detach().size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


def load_checkpoint(cfg_name: str, device: torch.device):
    net = DualStreamUNet()
    net.to(device)

    save_file = f'{cfg_name}'
    checkpoint = torch.load(save_file, map_location=device)

    net.load_state_dict(checkpoint['network'])

    return net


# @title
# dataset for classifying a scene
class SceneInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, s1: np.ndarray, s2: np.ndarray, tile_size: int = 256):
        super().__init__()

        self.tile_size = tile_size
        self.s1 = s1
        self.s2 = s2
        m, n, _ = s1.shape
        self.m = m // tile_size * tile_size
        self.n = n // tile_size * tile_size
        self.samples = []
        for i in range(0, m - tile_size, tile_size):
          for j in range(0, n - tile_size, tile_size):
            self.samples.append((i, j))
        self.length = len(self.samples)

        m_diff = tile_size - m % tile_size
        n_diff = tile_size - n % tile_size

        self.s1 = np.pad(self.s1, ((tile_size, m_diff), (tile_size, n_diff), (0, 0)), mode='reflect')
        self.s2 = np.pad(self.s2, ((tile_size, m_diff), (tile_size, n_diff), (0, 0)), mode='reflect')

    def __getitem__(self, index):
        # loading metadata of sample
        i, j = self.samples[index]

        i_min, j_min = i, j
        i_max, j_max = i + 3 * self.tile_size, j + 3 * self.tile_size

        s1_tile = TF.to_tensor(self.s1[i_min:i_max, j_min:j_max])
        s2_tile = TF.to_tensor(self.s2[i_min:i_max, j_min:j_max])

        item = {
            'x_s1': s1_tile,
            'x_s2': s2_tile,
            'i': i,
            'j': j,
        }
        return item

    def get_arr(self, dtype=np.uint8):
        return np.zeros((self.m, self.n), dtype=dtype)

    def __len__(self):
        return self.length

