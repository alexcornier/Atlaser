import sys

import os

import logging

from logging.handlers import RotatingFileHandler

import numpy as np

import pandas as pd

from PyQt5 import QtCore, QtWidgets, QtGui, Qt

import pyqtgraph.opengl as gl

import pyqtgraph as pg

from PIL import Image

from PIL import ImageEnhance

import json

import cv2

from pathlib import Path

from controls import TreeModel, LabeledSlider, EditViewBox, Transform, SliceImage, LabeledCircleWidget

from atlas import read_ontology, id_colors, color_atlas, get_atlas

import csv

from collections import defaultdict

# try:

#     import javabridge

#     import bioformats as bf

# except ImportError:

#     print('Bioformat not installed. Will not be able to open slide scanner files. '

#           'See https://pythonhosted.org/python-bioformats/ for installation instructions')





class Viewer(QtWidgets.QMainWindow):

    def __init__(self):

        # Create and initialize window

        super(Viewer, self).__init__()  # Calling the parent class __init__ method

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)



        # Logging

        self._logger = logging.getLogger('Atlaslog')

        self._logger.setLevel(logging.DEBUG)

        self._log_handler = RotatingFileHandler('Atlaser.log', maxBytes=1e6, backupCount=1)

        formatter = logging.Formatter(

            '%(asctime)s :: %(filename)s :: %(funcName)s :: line %(lineno)d :: %(levelname)s :: %(message)s',

            datefmt='%Y-%m-%d:%H:%M:%S')

        self._log_handler.setFormatter(formatter)

        self._logger.addHandler(self._log_handler)

        sys.excepthook = handle_exception



        self._logger.info('Initializing the window')

        # Menu bar

        self.file_menu = QtWidgets.QMenu('&File', self)

        self.file_menu.addAction('&Open data', self.select_data_file, QtCore.Qt.CTRL + QtCore.Qt.Key_O)

        # self.file_menu.addAction('&Save workspace', self.save_transf, QtCore.Qt.CTRL + QtCore.Qt.Key_S)

        # self.file_menu.addAction('&Load workspace', self.load_transf, QtCore.Qt.CTRL + QtCore.Qt.Key_L)

        self.file_menu.addAction('&Export points', self.export_points, QtCore.Qt.CTRL + QtCore.Qt.Key_E)

        self.file_menu.addAction('&Quit', self.close, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)

        self.edit_menu = QtWidgets.QMenu('&Edit', self)

        self.edit_menu.addAction('&Undo', self.undo, QtCore.Qt.CTRL + QtCore.Qt.Key_Z)

        # self.edit_menu.addAction('&Clear transformation', self.clear_transf, QtCore.Qt.CTRL + QtCore.Qt.ALT + QtCore.Qt.Key_C)

        self.edit_menu.addAction('Clear cells', self.clear_cells)
        
        self.help_menu = QtWidgets.QMenu('&Help', self)

        self.help_menu.addAction('&Manuel', self.help, QtCore.Qt.CTRL + QtCore.Qt.Key_H)

        self.menuBar().addMenu(self.file_menu)

        self.menuBar().addMenu(self.edit_menu)

        self.menuBar().addMenu(self.help_menu)



        # MAIN LAYOUT

        self.main_layout = QtWidgets.QHBoxLayout()



        # create a main widget

        window_size = [1000, 1500]

        self.main_widget = QtWidgets.QWidget(self)

        self.main_widget.setLayout(self.main_layout)

        self.main_widget.setGeometry(0, 0, window_size[0], window_size[1])

        self.main_widget.setWindowTitle('Slice viewer - No Data')

        self.setCentralWidget(self.main_widget)



        # LAYOUTS

	    #sépare ecran noir et menu scroll
        self.h_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        self.right_widget = QtWidgets.QWidget(self)

        self.g_layout = pg.GraphicsLayoutWidget(self.main_widget)

        self.v_layout_right = QtWidgets.QVBoxLayout()

        self.h_layout_right = QtWidgets.QHBoxLayout()

        self.h_layout_buttons = QtWidgets.QHBoxLayout()

        self.h_atlas = QtWidgets.QHBoxLayout()



        # WIDGETS



        # Slider for z and c -stack and zoom level

        #self.z_sl = LabeledSlider('Z slice')

        #self.z_sl.setSingleStep(1)

        #self.z_sl.setEnabled(False)

        #self.z_sl.valueChanged.connect(self.z_change)

        self.channel_sl = LabeledSlider('Channel')

        self.channel_sl.valueChanged.connect(self.channel_change)

        self.channel_sl.setSingleStep(1)

        self.channel_sl.setEnabled(False)

        # self.zoom_sl = LabeledSlider('Resolution level')

        # self.zoom_sl.valueChanged.connect(self.zoom_change)

        # self.zoom_sl.setSingleStep(1)

        # self.zoom_sl.setRange(0, 4)

        # self.zoom_sl.setEnabled(False)

        self.brain_sl = LabeledSlider('Brain slice')

        self.brain_sl.valueChanged.connect(self.brain_change)

        self.brain_sl.setSingleStep(1)

        self.brain_sl.setRange(0, 4)

        self.brain_sl.setEnabled(False)

        # Sliders for luminosity and contrast

        self.lum_sl = LabeledSlider('Brightness')

        self.lum_sl.setRange(0, 100)

        self.lum_sl.setValue(50)

        self.lum_sl.valueChanged.connect(self.luminosity_change)

        self.lum_sl.setEnabled(False)

        self.contrast_sl = LabeledSlider('Contrast')

        self.contrast_sl.setRange(0, 100)

        self.contrast_sl.setValue(50)

        self.contrast_sl.valueChanged.connect(self.contrast_change)

        self.contrast_sl.setEnabled(False)

        # Cell selection mode checkbox

        self.cell_select_cb = QtWidgets.QCheckBox('Cell selection mode')

        self.cell_select_cb.setChecked(False)

        self.cell_select_cb.clicked.connect(self.switch_cellmode)

        # Slider for atlas slice

        self.slice_sl = LabeledSlider('Atlas slice')

        self.slice_sl.setRange(0, 1)

        self.slice_sl.setSingleStep(1)

        self.slice_sl.valueChanged.connect(self.update_atlas)

        self.alpha_sl = LabeledSlider('Atlas opacity',)

        self.alpha_sl.setRange(0, 100)

        self.alpha_sl.setSingleStep(1)

        self.alpha_sl.setValue(50)

        self.alpha_sl.valueChanged.connect(self.atlas_alpha)

        # self.trans_sl = LabeledSlider('Translation')

        # Transformation sliders
        
        # Circle widget to rotate the image
        self.rot_sl = LabeledCircleWidget('Image rotation', factor = 1)

        self.rot_sl.setRange(0, 360)

        self.rot_sl.setSingleStep(1)

        self.rot_sl.valueChanged.connect(self.sl_rot_changed)

        self.rot_sl.setValue(90)

        self.scale_sl = LabeledSlider('Scale', factor = 1000)

        self.scale_sl.setRange(1, 2000)

        self.scale_sl.setSingleStep(1)

        self.scale_sl.valueChanged.connect(self.sl_scale_changed)

        # Show/Hide atlas check box

        self.show_atlas_cb = QtWidgets.QCheckBox('Show atlas')

        self.show_atlas_cb.setChecked(True)

        self.show_atlas_cb.stateChanged.connect(self.show_atlas)

        # Atlas orientation combo box

        self.orientation_cb = QtWidgets.QComboBox(self)

        self.orientation_cb.addItems(['Coronal', 'Horizontal', 'Sagittal'])

        self.orientation_cb.currentIndexChanged.connect(self.change_orientation)

        # Region tree

        self.tree = QtWidgets.QTreeView(self.right_widget)

        self.onto = read_ontology('mouse_ontology.json')

        self.tree_model = TreeModel(None, self.onto)

        self.tree.setModel(self.tree_model)

        self.tree.expanded.connect(self.expanded)

        self.tree.selectionModel().currentChanged.connect(self.select_region)

        # Buttons

        # self.register_pb = QtWidgets.QPushButton('&Register')

        self.fliplr_pb = QtWidgets.QPushButton('Flip &left-right')

        self.fliplr_pb.setCheckable(True)

        self.flipud_pb = QtWidgets.QPushButton('Flip &up-down')

        self.flipud_pb.setCheckable(True)

        self.fliplr_pb.clicked.connect(self.flip_lr)

        self.flipud_pb.clicked.connect(self.flip_ud)

        # self.register_pb.clicked.connect(self.align)

        # Graph widgets

        self.anat_image = pg.ImageItem()

        self.atlas_image = pg.ImageItem()

        self.template_image = pg.ImageItem()

        self.zoom_image = pg.ImageItem()

        # self.anat_image.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)

        self.atlas_image.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)

        # Mouse sensitivity on both windows

        self.atlas_image.setZValue(10)

        self.zoom_image.setZValue(10)

        # Create the selection tool (circles)

        self.cell_scatter = pg.ScatterPlotItem()

        self.cell_pen = pg.mkPen(color=(242, 142, 85, 200), width=1.5)

        self.cell_brush = pg.mkBrush(None)

        self.cell_scatter.setBrush(self.cell_brush)

        self.cell_scatter.setPen(self.cell_pen)

        # Creation of ViewBox anat : adding the atlas and the image

        self.vb_anat = EditViewBox()

        self.vb_anat.rotation.connect(self.rotation)

        self.vb_anat.translation.connect(self.translation)

        self.vb_anat.scale.connect(self.scaling)

        self.vb_anat.cell_select.connect(self.cell_clicked)

        self.g_layout.addItem(self.vb_anat, row=0, col=0)

        self.vb_anat.setAspectLocked()

        self.vb_anat.invertY()

        # Ajoute l'image à l'atlas = ils ne font plus qu'un

        self.vb_anat.addItem(self.anat_image)

        self.vb_anat.addItem(self.atlas_image)

        self.vb_anat.addItem(self.cell_scatter)

        # self.vb_atlas = self.g_layout.addViewBox(row=1, col=0)

        # self.vb_atlas.setAspectLocked()

        # self.vb_atlas.invertY()

        # self.vb_atlas.addItem(self.template_image)

        # self.vb_anat.setXLink(self.vb_atlas)

        # self.vb_anat.setYLink(self.vb_atlas)

        # Zoom inset on the high quality image

        self.vb_inset = self.g_layout.addViewBox(row=0, col=1)

        self.vb_inset.addItem(self.zoom_image)

        self.vb_inset.invertY()

        self.vb_inset.setAspectLocked(True)
	
	    # Modification du pointeur

        cursor_pen = pg.mkPen(color=(242, 142, 85), width=2)

        self._h_line_l = QtWidgets.QGraphicsLineItem(0, 2, 2, 2)

        self._h_line_r = QtWidgets.QGraphicsLineItem(1, 2, 4, 2)

        self._v_line_u = QtWidgets.QGraphicsLineItem(2, 3, 2, 4)

        self._v_line_d = QtWidgets.QGraphicsLineItem(2, 0, 2, 3)

        self._h_line_l.setParentItem(self.zoom_image)

        self._h_line_r.setParentItem(self.zoom_image)

        self._v_line_u.setParentItem(self.zoom_image)

        self._v_line_d.setParentItem(self.zoom_image)

        self._h_line_l.setPen(cursor_pen)

        self._h_line_r.setPen(cursor_pen)

        self._v_line_u.setPen(cursor_pen)

        self._v_line_d.setPen(cursor_pen)

        self.vb_inset.enableAutoRange()

        # self.vb_atlas.addItem(self._v_line)

        # self.vb_atlas.addItem(self._h_line)


        # LAYOUT setup

        self.right_widget.setLayout(self.v_layout_right)

        self.v_layout_right.addWidget(self.tree)

        # self.v_layout_right.addWidget(self.z_sl)

        self.v_layout_right.addWidget(self.channel_sl)

        # self.v_layout_right.addWidget(self.zoom_sl)

        self.v_layout_right.addWidget(self.brain_sl)

        self.v_layout_right.addWidget(self.lum_sl)

        self.v_layout_right.addWidget(self.contrast_sl)

        #self.v_layout_right.addWidget(self.image_zoom)

        self.h_atlas.addWidget(self.orientation_cb)

        self.h_atlas.addWidget(self.show_atlas_cb)

        self.v_layout_right.addLayout(self.h_atlas)

        self.v_layout_right.addWidget(self.slice_sl)

        self.v_layout_right.addWidget(self.alpha_sl)

        self.v_layout_right.addWidget(self.rot_sl)

        # Plus besoin d'afficher le widget pour modifier la résolution étant donné que la valeur est fixée à 1
        # self.v_layout_right.addWidget(self.scale_sl)

        self.h_layout_buttons.addWidget(self.fliplr_pb)

        self.h_layout_buttons.addWidget(self.flipud_pb)

        self.v_layout_right.addLayout(self.h_layout_buttons)

        self.v_layout_right.addWidget(self.cell_select_cb)

        # self.h_layout_right.addWidget(self.register_pb)

        self.v_layout_right.addLayout(self.h_layout_right)

        self.h_splitter.addWidget(self.g_layout)

        self.h_splitter.setStretchFactor(0, 5)

        self.h_splitter.addWidget(self.right_widget)

        self.main_layout.addWidget(self.h_splitter)



        # Some shortcuts
        
        # Déplacement de slice en slice avec les flèches gauche et droite

        atlas_fwd = QtWidgets.QShortcut(QtCore.Qt.Key_Right, self.main_widget)

        atlas_fwd.activated.connect(self.next_slice)

        atlas_bck = QtWidgets.QShortcut(QtCore.Qt.Key_Left, self.main_widget)

        atlas_bck.activated.connect(self.prev_slice)

        self._logger.info('End of window creation')



    def switch_cellmode(self, checked):

        pass



    def export_points(self):

        pass



    def zoom_change(self, value):

        pass



    def atlas_alpha(self, value):

        pass



    def undo(self):

        pass



    def show_atlas(self):

        pass



    def channel_change(self, value):

        pass



    def z_change(self, value):

        pass



    def brain_change(self, value):

        pass



    def luminosity_change(self, value):

        pass



    def contrast_change(self, value):

        pass



    def next_slice(self):

        pass



    def prev_slice(self):

        pass



    def flip_lr(self):

        pass



    def flip_ud(self):

        pass



    def select_region(self, current, previous):

        pass



    def expanded(self, index):

        self.tree.resizeColumnToContents(1)

        self.tree.resizeColumnToContents(0)



    def rotation(self, value):

        pass



    def translation(self, x_shift, y_shift):

        pass



    def scaling(self, scale):

        pass



    def update_atlas(self, value):

        pass



    def change_orientation(self, index):

        pass



    def select_data_file(self):

        pass



    def sl_rot_changed(self, value):

        pass



    def sl_scale_changed(self, value):

        pass



    def save_transf(self):

        pass



    def load_transf(self):

        pass



    def align(self):

        pass



    def cell_clicked(self, x, y, mx, my):

        pass



    def clear_transf(self):

        pass



    def clear_cells(self):

        pass


    def help_menu(self):

        pass





class AtlasExplorer(Viewer):

    def __init__(self):

        super().__init__()

        self.statusBar().showMessage('Loading...')

        self._logger.info('Loading...')

        Image.MAX_IMAGE_PIXELS = None   # Necessary to open super large tiff files

        # Load atlas and annotations

        self.atlas = np.load('contourify2.npy')

        self.template = get_atlas('average_template_25.nrrd')

        self.raw_atlas = get_atlas('annotation_25.nrrd')

        self.colors = id_colors(self.onto)

        self.c_atlas = np.load('color_atlas.npy')

        self._logger.info('Atlas opened')

        self.p_max = 2**16

        # Change the number value to open by default a specific atlas (0: coronal, 1: horizontal, 2: sagittal)
        self._c_orient = 0

        self._sel_regions = {}

        self.orientation_cb.setCurrentIndex(2)

        self._data_path = None

        self.slice_image = SliceImage('')

        self.pic = np.zeros((5, 5), dtype=np.uint8)

        self._transf_raw = Image.new('L', (5, 5))   # Create a greyscale 8-bit image. Size of 5x5

        self.transf_raw = self._transf_raw

        # Modifie la taille de la croix à l'ouverture, mais se réinitialise à l'ouverture de l'image.
        # Néanmoins, la fenêtre est plus petite, mais la sélection n'est plus la même (ne pointe plus sur la même chose)
        self._raw_inset = np.zeros((5, 5), dtype=np.uint8)

        self.raw_inset = self._raw_inset

        self._transf = Transform()

        self.transf = self._transf

	    # Modifie le niveau de zoom de l'image de droite : plus tu augmentes, plus le zoom baisse (plus de pixel à l'écran donc moins zoomé)

        self._inset_size = 1000

        self.cells = []

        self.cell_pos = []

        self.actions = []

        self.rot_sl.setValue(90)

        self.scale_sl.setValue(int(self.transf.scale * self.scale_sl.factor))

        # Capture mouse movements to scroll the inset

        self._proxy = pg.SignalProxy(self.vb_anat.scene().sigMouseMoved, rateLimit=0.1, delay=.1, slot=self.mouse_moved)

        self.setWindowTitle('Atlaser Sotfware')

        self.help_menu = None




    def switch_cellmode(self, checked):

        if checked:

            self.raw_inset = np.zeros((5, 5), dtype=np.uint8)

            self.transf_raw = self.apply_transf(self.slice_image.raw_img, False, False)

            self._proxy.rateLimit = 30

            self.statusBar().showMessage('Ready for cell selection', 1500)



    def draw_cross(self):

        """

        Draw the cross cursor on the inset zoom picture

        Place it at the center of the inset, which depends on the exact size of the self._raw_inset array

        """

        # inset size

        w, h = self._raw_inset.shape

        w2, h2 = w//2, h//2

        # Cross on the inset picture

        self._h_line_l.setLine(0, h2, w2, h2)

        self._h_line_r.setLine(w2, h2, w, h2)

        self._v_line_u.setLine(w2, h2, w2, h)

        self._v_line_d.setLine(w2, h2, w2, 0)



    def clear_cells(self):

        self.cells = []

        self.cell_pos = []

        self.cell_scatter.setData(self.cell_pos)



    def clear_transf(self):

        self.transf = Transform()


    def mouse_moved(self, pos):

        """

        Update the zoom inset from the raw image when the mouse cursor moves over the low resolution atlas aligned image


        Parameters

        ----------

        pos: list of SceneEvents

            Contains the mouse position (in scene coordinates, will need conversion)

        """

        # FIXME: When too close to the edges, not possible to get a precise position
        # Ne marche pas car :
            # Zone de clic = zone de l'atlas
            # Si on sort la souris de l'atlas, ce n'est plus précis car on ne peut plus cliquer dans tous les cas

        if not self.cell_select_cb.isChecked():

            return

        pos = pos[0]

        v_range = self.vb_anat.viewRange()

        x, y = self.convert_mouse_pos(pos.x(), pos.y(), v_range[0][0], v_range[1][0])

        x, y = x, y

        all_scales = self.slice_image.downfactor / self.transf.scale

        raw_x = int((x - self.transf.translation[1]) * all_scales)

        raw_y = int((y - self.transf.translation[0]) * all_scales)

        # If the mouse is out of the picture, don't bother

        if raw_x > self.transf_raw.height or raw_x < 0 or raw_y > self.transf_raw.width or raw_y < 0:

            return

        if self.transf_raw.width < 10:

            return

        raw_x, raw_y = raw_y, raw_x

        start_x = np.clip(raw_x - self._inset_size, 0, self.transf_raw.width - self._inset_size)

        start_y = np.clip(raw_y - self._inset_size, 0, self.transf_raw.height - self._inset_size)

        stop_x = np.clip(raw_x + self._inset_size, self._inset_size, self.transf_raw.width - 1)

        stop_y = np.clip(raw_y + self._inset_size, self._inset_size, self.transf_raw.height - 1)

        self.raw_inset = np.array(self.transf_raw.crop((start_x, start_y, stop_x, stop_y)))

        self.draw_cross()



    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:

        """

        closeEvent of Viewer.

        Perform some cleanup

        """

        # if self.bioformat:

        #     javabridge.kill_vm()

        #     self._logger.info('Kill the Java VM')

        a0.accept()

        super().closeEvent(a0)



    def export_points(self):
        try : 
            for i in range(0, len(self.cells)):

                if (self.cells[i]['Region'] == 'None'):

                    d = defaultdict(str)

                    d[1] = ""

                    d[2] = ""

                    d = dict(d)

                    self.cells[i].update(d)

                    self.cells[i]['structure'] = self.cells[i].pop('structure')

                    del self.cells[i]['Region']
                    

                else:

                    d = defaultdict(str)

                    d[1] = self.cells[i]['Region'][len(self.cells[i]['Region']) - 2]

                    d[2] = self.cells[i]['Region'][len(self.cells[i]['Region']) - 3]

                    d = dict(d)

                    self.cells[i].update(d)

                    self.cells[i]['structure'] = self.cells[i].pop('structure')

                    del self.cells[i]['Region']
                    

            im_path = Path(self.data_path)

            path = im_path.parent / f'cells_{im_path.stem}.csv'

            keys = self.cells[i].keys()

            with open(path, 'a', newline='') as file:

                dict_writer = csv.DictWriter(file, keys)

                dict_writer.writerows(self.cells)

            self.cells = []
        
        
        except KeyError:

            pass

            

    


    # Displays the image in the right windows when cell mode selection is activated
    
    def apply_transf(self, img, scale=True, trans=True):

        new_img = img.copy()

        if self.transf.flipped_lr:

            new_img = Image.fromarray(np.flipud(np.array(new_img)))

        if self.transf.flipped_ud:

            new_img = Image.fromarray(np.fliplr(np.array(new_img)))

        n_width = int(new_img.width*self.transf.scale)

        n_height = int(new_img.height * self.transf.scale)

        if scale:

            new_img = new_img.resize((n_width, n_height))

        if self.transf.rotation != 90:

            new_img = new_img.rotate(self.transf.rotation, expand=True)

        else:

            new_img = Image.fromarray(np.flipud(np.array(new_img).T))

        w, h = new_img.size

        if trans:

            x_shift, y_shift = self.transf.translation

        else:

            x_shift, y_shift = 0, 0

        buffer_img = Image.new(img.mode, (w+abs(x_shift), h+abs(y_shift)))

        buffer_img.paste(new_img, (x_shift, y_shift))

        return buffer_img



    def update_img(self):

        if self._data_path is None:

            return

        # self.slice_image.img = self.apply_transf(self.slice_image.dw_img)

        self.pic = np.array(self.apply_transf(self.slice_image.dw_img))

        # The old transformation is no longer valid for the inset display and cell selection

        self.cell_select_cb.setChecked(False)

        # Erase the previously transformed image to avoid confusion

        # Now the transformed picture needs to be recomputed by switching into cell selection mode

        self.raw_inset = np.zeros((5, 5), dtype=np.uint8)

        self.transf_raw = Image.new('L', (5, 5))

        # Useless to keep track of mouse movements too often since we have nothing to display

        self._proxy.rateLimit = 0.1



    def convert_mouse_pos(self, x, y, min_x, min_y):

        # Original position zoom corrected

        ex, ey = x, y

        # Correct for zoom

        px_width, px_height = self.anat_image.pixelSize()

        x /= px_width

        y /= px_height

        # Inverse transformation from display to data

        coords = np.intp(pg.transformCoordinates(self.anat_image.inverseDataTransform(), np.array([x, y])))

        dx = min_x + coords[0]

        dy = min_y + coords[1]

        # Divide dx and dy by the number of the atlas scale (here, 10)

        dx = dx

        dy = dy

        return dx, dy



    def cell_clicked(self, x, y, mx, my):

        dx, dy = self.convert_mouse_pos(x, y, mx, my)

        dx, dy = int(dx), int(dy)

        # Get structure id if possible

        c_atlas = self.get_slice(self.raw_atlas, self.c_slice)

        if dx < 0 or dx >= c_atlas.shape[0] or dy < 0 or dy >= c_atlas.shape[1]:

            return

        reg_id = c_atlas[dx, dy]

        # Get region name

        l_reg = [(i, r) for i, r in enumerate(self.tree_model.regions) if r.id == reg_id]

        if len(l_reg) == 0:

            return

        else:

            l_reg = l_reg[0] # id de la structure récupérée
        
        reg = l_reg[1]  # nom de la structure récupérée

        parent_reg = get_parent(self.onto, l_reg[0])

        if parent_reg == None:

            parent_reg = str(None)

        reg_ix = self.tree_model.match(self.tree_model.index(0, 0, QtCore.QModelIndex()), QtCore.Qt.DisplayRole, QtCore.QVariant(reg.abbr),

                                    1, QtCore.Qt.MatchExactly|QtCore.Qt.MatchRecursive)

        if reg_ix:

            reg_ix = reg_ix[0]

        self.tree.setCurrentIndex(reg_ix)

        if self.cell_select_cb.isChecked():

            #setOpacity(laValeurDeLaCase)

            self.cells.append({'pos': (dx + 1, dy + 1), 'Region': parent_reg, 'structure': str(reg)})

            self.cell_pos.append((dx + 1, dy + 1))

            self.actions.append(((self.cells.pop, self.cell_pos.pop), (-1, -1)))

            self.cells = check_duplicate(self.cells)
    
            self.cell_scatter.setData(pos = self.cell_pos)

            self._logger.debug(f'Coordonnées enregistrées: {self.cells[0]}')

            for i in range(0, len(self.cell_pos) - 1):

                if (dx + 1, dy + 1) == self.cell_pos[i]:

                    self.cell_pos.remove(self.cell_pos[i])

                    self.cell_scatter.setData(pos = self.cell_pos)
            
            for i in range(0, len(self.cells) - 1):

                if (dx + 1, dy + 1) == self.cells[i]['pos']:
                    
                    self.cells.remove(self.cells[i])

                    self._logger.debug("Supression du point dans self.cells réalisée")





    def zoom_change(self, value):

        self.slice_image.c_zoom = value

        self.auto_scale()



    def atlas_alpha(self, value):

        self.update_atlas(self.slice_sl.value())



    def undo(self):

        actions, args = self.actions.pop(-1)

        for a, ix in zip(actions, args):

            a(ix)
        
        self.cell_pos = check_duplicate(self.cell_pos)

        self.cell_scatter.setData(pos = self.cell_pos)



    def show_atlas(self):

        self.atlas_image.setVisible(self.show_atlas_cb.isChecked())
        


    def channel_change(self, value):

        self.slice_image.c_channel = value

        self.auto_scale()



    def z_change(self, value):

        self.slice_image.c_zslice = value

        self.auto_scale()



    def brain_change(self, value):

        self.slice_image.c_slice = value

        self._logger.debug('Changing brain slice')

        self.auto_scale()



    def luminosity_change(self, value):

        self.apply_brightness(value)



    def contrast_change(self, value):

        self.pic = self.apply_contrast(self.slice_image.img, value)



    def apply_brightness(self, value):

        self.anat_image.setLevels((0, self.slice_image.p_max * (2 - value / 50)))



    @staticmethod

    def apply_contrast(img, value):

        def contrast_lut(v):

            m = 2**16   # Max in 16 bits

            v /= m      # Normalize to 1

            # 50 is the middle of the slider, so no extra contrast.

            # Over 50: exponent from 1 to 2

            # Under 50: exponent from 0 to 1

            pv = np.clip(np.power(v,  value / 50), 0, 1)

            pv *= m     # Go back to original scale

            return pv



        pic = np.array(img, dtype=np.float32)

        return contrast_lut(pic)



    def next_slice(self):

        self.slice_sl.setValue(self.c_slice + 1)

        self.apply_brightness(self.lum_sl.value())




    def prev_slice(self):

        self.slice_sl.setValue(self.c_slice - 1)

        self.apply_brightness(self.lum_sl.value())


    def flip_lr(self):

        self.transf.flipped_lr = not self.transf.flipped_lr

        self.apply_brightness(self.lum_sl.value())



    def flip_ud(self):

        self.transf.flipped_ud = not self.transf.flipped_ud

        self.apply_brightness(self.lum_sl.value())



    def rotation(self, value):

        self.transf.add_rotation(value)

        self.update_transf_sliders()



    def translation(self, x_shift, y_shift):

        x_shift, y_shift = int(-x_shift), int(-y_shift)

        self.transf.add_translation((x_shift, y_shift))

        self.apply_brightness(self.lum_sl.value())



    def scaling(self, value):

        scale = value / self.slice_image.img.height / 5

        self.transf.add_scale(scale)

        self.update_transf_sliders()

        self.apply_brightness(self.lum_sl.value())



    def sl_rot_changed(self, value):

        self.transf.rotation = value / self.rot_sl.factor

        self.apply_brightness(self.lum_sl.value())




    # def sl_scale_changed(self, value):

    #     if value == 1:

    #         return

    #     self.transf.scale = value / self.scale_sl.factor



    def update_transf_sliders(self):

        self.rot_sl.setValue(int(self.transf.rotation))# * self.rot_sl.factor))

        # self.scale_sl.setValue(int(self.transf.scale * self.scale_sl.factor))

        self.fliplr_pb.setChecked(bool(self.transf.flipped_lr))

        self.flipud_pb.setChecked(bool(self.transf.flipped_ud))



    @property

    def transf_raw(self):

        return self._transf_raw



    @transf_raw.setter

    def transf_raw(self, value):

        self._transf_raw = value

        self.raw_inset = np.zeros((5, 5), dtype=np.uint8)



    @property

    def raw_inset(self):

        return self._raw_inset



    @raw_inset.setter

    def raw_inset(self, value):

        self._raw_inset = value

        self.zoom_image.setImage(value)



    @property

    def transf(self):

        return self._transf



    @transf.setter

    def transf(self, value):

        self._transf = value

        self.transf.rotation_changed.connect(self.update_img)

        self.transf.translation_changed.connect(self.update_img)

        self.transf.scale_changed.connect(self.update_img)

        self.transf.fliplr_changed.connect(self.update_img)

        self.transf.flipud_changed.connect(self.update_img)

        self.update_img()

        self.update_transf_sliders()



    @property

    def sel_regions(self):

        return self._sel_regions



    @sel_regions.setter

    def sel_regions(self, value):

        self._sel_regions = value

        self.update_atlas(self.slice_sl.value())



    @property

    def pic(self):

        return self._pic



    @pic.setter

    def pic(self, value):

        self._pic = value

        self.anat_image.setImage(value)



    @property

    def data_path(self):

        return self._data_path



    @data_path.setter

    def data_path(self, value):

        """

        When the data path changes, open the corresponding data file



        Parameters

        ----------

        value: str

            Path to the data as selected from the dialog box

        """

        self._data_path = value

        self.load_image()



    @property

    def c_orient(self):

        return self._c_orient



    @c_orient.setter

    def c_orient(self, value):

        self._c_orient = value

        self.slice_sl.setRange(0, self.template.shape[value] - 1)



    def get_slice(self, volume, value):

        if self.c_orient == 0:

            s = volume[value, ...]

            d = (1, 0, 2) if len(s.shape) == 3 else (1, 0)

            s = s.transpose(d)

        elif self.c_orient == 1:

            s = volume[:, value, ...]

        else:

            if len(volume.shape) == 3:

                s = volume[:, :, value]

            else:

                s = volume[:, :, value, :]

        return s



    def update_atlas(self, value):

        self.c_slice = value

        c_atlas = self.get_slice(self.c_atlas, value)

        s_atlas = self.get_slice(self.atlas, value)

        c_template = self.get_slice(self.template, value)

        atlas_alpha = int(self.alpha_sl.value()/100*255)

        ids = np.unique(s_atlas)

        alpha = np.zeros(s_atlas.shape) + atlas_alpha

        alpha[np.isin(s_atlas, list(self.sel_regions))] = 0

        gi = np.logical_and(alpha == atlas_alpha, np.any(c_atlas > 0, 2))

        c_atlas[gi, ...] = 255

        c_atlas = np.dstack((c_atlas, alpha))

        self.template_image.setImage(c_template)

        self.atlas_image.setImage(c_atlas)



    def change_orientation(self, index):

        self.c_orient = index

        self.update_atlas(self.slice_sl.value())



    def select_data_file(self):

        cwd = os.getcwd()

        dpath, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "Choose an image file",

                                                               cwd, 'Image (*.tiff *.tif *.vsi *.ndpis)')

        if dpath != '':

            # Set the new data path. The data_path setter will take care of the rest

            self.data_path = dpath



    # Helping manuel to get shortcuts and tips

    def help(self):
        
        img = cv2.imread("helpManuel.png")
        
        cv2.imshow('Help manuel', img)



    def auto_scale(self):

        atlas_shape = self.c_atlas.shape[:2]

        self.actions = []

        self.cells = []

        self.cell_pos = []

        # On fixe le scale à 1 (et on ne transforme plus cette valeur comme ci-dessous)
        self.transf.scale = 1

        # if self.transf.scale != 1:

        #     self.transf.scale = min(atlas_shape[0] / self.slice_image.img.width,

        #                             atlas_shape[1] / self.slice_image.img.height)

        #     self.update_transf_sliders()

        self.update_img()



    def update_n_slices(self, n_zslices, n_channels, n_res_levels, n_brainslices):

        # self.z_sl.setRange(0, n_zslices - 1)

        self.channel_sl.setRange(0, n_channels - 1)

        # self.zoom_sl.setRange(0, n_res_levels - 1)

        self.brain_sl.setRange(0, n_brainslices - 1)

        self._logger.debug(f'Number of z slices: {n_zslices}')



    def update_max_value(self, p_max):

        self.p_max = p_max

        self.apply_brightness(self.lum_sl.value())



    def img_updated_event(self):

        self.pic = self.apply_contrast(self.slice_image.img, self.contrast_sl.value())

        self.apply_brightness(self.lum_sl.value())

        self._logger.debug(f'Pic max: {self.pic.max()}')

        self._logger.debug(f'Display range: {self.anat_image.levels}')



    def load_image(self):

        """

        Load the image defined in self.data_path

        Loading procedure depends on format. Handles tif through PIL, vsi through python-bioformats

        """

        # Load the imge

        self.slice_image = SliceImage(self.data_path)

        self.slice_image.n_slices_known.connect(self.update_n_slices)

        self.slice_image.img_updated.connect(self.img_updated_event)

        self.slice_image.max_value_changed.connect(self.update_max_value)

        self.slice_image.load()



        # Reset contrast and brightness

        # self.contrast_sl.setValue(50)

        # self.lum_sl.setValue(50)

        if self.slice_image.is_tiff:

            # self.z_sl.setEnabled(False)

            self.channel_sl.setEnabled(False)

            self.scale_sl.setValue(0.2)

        else:

            # self.z_sl.setEnabled(True)

            # self.zoom_sl.setEnabled(True)

            self.channel_sl.setEnabled(True)

            # self.z_sl.setValue(self.slice_image.c_zslice)

            self.channel_sl.setValue(self.slice_image.c_channel)

            # self.zoom_sl.setValue(self.slice_image.c_zoom)

        if self.slice_image.is_ndpis:

            self.brain_sl.setEnabled(True)

        else:

            self.brain_sl.setEnabled(False)

    # Finish initialization

        self.auto_scale()

        self.lum_sl.setEnabled(True)

        self.contrast_sl.setEnabled(True)

        self.transf_raw = self.slice_image.raw_img



    def select_region(self, current, previous):

        c_item = self.tree.model().data(current, QtCore.Qt.UserRole)

        sel_regions = set([c_item.region.id])

        children = set([ch.region.id for ch in self.get_all_children(c_item)])

        sel_regions.update(children)

        self.sel_regions = sel_regions

        self._logger.debug('Region selected')



    def get_all_children(self, region):

        for ch in region.child_items:

            yield ch

            if ch.child_items:

                yield from self.get_all_children(ch)



    def save_transf(self):

        if self.data_path is None:

            return

        self._logger.debug(self.cells)

        params = dict(manual=self.transf.params, image=self.data_path, cells=self.cells)

        params['manual']['atlas_orientation'] = self.c_orient

        params['manual']['atlas_slice'] = self.slice_sl.value()

        params['manual']['prefactor'] = self.slice_image.downfactor

        im_path = Path(self.data_path)

        path = im_path.parent / f'params_{im_path.stem}.json'

        with open(path, 'w') as f:

            json.dump(params, f, indent=4)



    def load_transf(self):

        cwd = os.getcwd()

        dpath, _filter = QtWidgets.QFileDialog.getOpenFileName(self, "Choose a transformation file",

                                                               cwd, 'JSON (*.json)')

        if dpath != '':

            # Load the transformation

            with open(dpath, 'r') as f:

                workspace = json.load(f)

                image = workspace['image']

                d_transf = workspace['manual']

                l_cells = workspace['cells']

            transf = Transform((d_transf['x_shift'], d_transf['y_shift']), d_transf['rotation'],

                               d_transf['scale'], d_transf['flip_lr'], d_transf['flip_ud'])

            try:

                self.orientation_cb.setCurrentIndex(d_transf['atlas_orientation'])

                self.slice_sl.setValue(int(d_transf['atlas_slice']))

            except KeyError:

                pass

            self.data_path = image

            self.transf = transf

            self.cells = l_cells

            self.cell_pos = [tuple(c['pos']) for c in l_cells]

            self.cell_scatter.setData(pos=self.cell_pos)



    # def align(self):

    #     c_pos = self.slice_sl.value()

    #     template = self.get_slice(self.template, slice(max(0, c_pos-3), min(c_pos+4, self.template.shape[-1])))

    #     template = template.transpose((0, 2, 1)).astype(self.pic.dtype)

    #

    #     new_img = self._raw_img.copy()

    #     new_img = new_img.rotate(self.transf.rotation, expand=True)

    #     w, h = new_img.size

    #     x_shift, y_shift = self.transf.translation

    #     buffer_img = Image.new(self._raw_img.mode, (w+abs(x_shift), h+abs(y_shift)))

    #     buffer_img.paste(new_img, (x_shift, y_shift))

    #     new_img.close()

    #     self.r_im = register(template, np.array(buffer_img).T, self.transf)

    #     buffer_img.close()

    #     self.anat_image.setImage(self.r_im)





def handle_exception(exc_type, exc_value, exc_traceback):

    """Handle uncaugt exceptions and print in logger."""

    logger = logging.getLogger('Atlaslog')

    if issubclass(exc_type, KeyboardInterrupt):

        sys.__excepthook__(exc_type, exc_value, exc_traceback)

        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))



def check_duplicate(list):

    res = [] 

    for i in list: 

        if i not in res: 

            res.append(i)

    return res 



def get_parent(json_tree, target_id):

    for element in json_tree:

        if element['id'] == target_id:

            return [element['id']]

        else:

            if element['children']:

                check_child = get_parent(element['children'], target_id)

                if check_child:

                    return [element['name']] + check_child




if __name__ == '__main__':

    qApp = QtWidgets.QApplication(sys.argv)

    window = AtlasExplorer()

    window.setWindowIcon(QtGui.QIcon("logo.png"))

    window.show()

    window.setGeometry(40, 20, 1000, 800)

    sys.exit(qApp.exec_())

