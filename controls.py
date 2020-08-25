from PyQt5 import QtWidgets, QtGui, QtCore

from pyqtgraph import ViewBox

import numpy as np

from math import sqrt

import logging

from pathlib import Path

from PIL import Image

import nanozoomer as nz

try:

    import javabridge

    import bioformats as bf

except ImportError:

    print('Bioformat not installed. Will not be able to open slide scanner files. '

          'See https://pythonhosted.org/python-bioformats/ for installation instructions')





HORIZONTAL_HEADERS = ('Acronym', 'Name', 'Color')





class Region(object):

    def __init__(self, id, name, abbr, color, hemisphere, parent_id, parent=None):

        self.id = id

        self.name = name

        self.abbr = abbr

        self.color = color

        self.hemisphere = hemisphere

        self.parent = parent

        self.data = (abbr, name, color)



    def __len__(self):

        return len(HORIZONTAL_HEADERS)



    def __repr__(self):

        return f'{self.abbr}: {self.name}'



    __str__ = __repr__





class TreeItem(object):

    def __init__(self, region, parent=None):

        self.parent_item = parent

        self.region = region

        self.child_items = []



    def appendChild(self, item):

        self.child_items.append(item)



    def child(self, row):

        return self.child_items[row]



    def childCount(self):

        return len(self.child_items)



    def columnCount(self):

        return len(self.region)



    def data(self, column):

        try:

            return QtCore.QVariant(self.region.data[column])

        except IndexError:

            return QtCore.QVariant()



    def parent(self):

        return self.parent_item



    def row(self):

        if self.parent_item:

            return self.parent_item.child_items.index(self)



        return 0



    def setData(self, data):

        self.region = data





class TreeModel(QtCore.QAbstractItemModel):

    """

    A model to display regions

    """

    def __init__(self, parent=None, onto=[]):

        super(TreeModel, self).__init__(parent)

        self.regions = []

        r = onto[0]

        region = Region(r['id'], r['name'], r['acronym'], r['color_hex_triplet'], r['hemisphere_id'],

                        r['parent_structure_id'])

        self.rootItem = TreeItem(region)

        self.regions.append(region)

        self.create_regions(r['children'], self.rootItem)

        self.parents = {0: self.rootItem}



    def create_regions(self, onto, parent=None):

        for r in onto:

            region = Region(r['id'], r['name'], r['acronym'], r['color_hex_triplet'], r['hemisphere_id'],

                            r['parent_structure_id'], parent)

            self.regions.append(region)

            new_item = TreeItem(region, parent)

            if parent is not None:

                parent.appendChild(new_item)

            self.create_regions(r['children'], new_item)



    def columnCount(self, parent=None):

        if parent and parent.isValid():

            return parent.internalPointer().columnCount()

        else:

            return len(HORIZONTAL_HEADERS)



    def data(self, index, role):

        if not index.isValid():

            return QtCore.QVariant()



        item = index.internalPointer()

        if role == QtCore.Qt.DisplayRole:

            return item.data(index.column())

        if role == QtCore.Qt.UserRole:

            if item:

                return item



        return QtCore.QVariant()



    def headerData(self, column, orientation, role):

        if (orientation == QtCore.Qt.Horizontal and

                role == QtCore.Qt.DisplayRole):

            try:

                return QtCore.QVariant(HORIZONTAL_HEADERS[column])

            except IndexError:

                pass



        return QtCore.QVariant()



    def index(self, row, column, parent):

        if not self.hasIndex(row, column, parent):

            return QtCore.QModelIndex()



        if not parent.isValid():

            parent_item = self.rootItem

        else:

            parent_item = parent.internalPointer()



        child_item = parent_item.child(row)

        if child_item:

            return self.createIndex(row, column, child_item)

        else:

            return QtCore.QModelIndex()



    def parent(self, index):

        if not index.isValid():

            return QtCore.QModelIndex()



        child_item = index.internalPointer()

        if not child_item:

            return QtCore.QModelIndex()



        parent_item = child_item.parent()



        if parent_item == self.rootItem:

            return QtCore.QModelIndex()



        return self.createIndex(parent_item.row(), 0, parent_item)



    def rowCount(self, parent=QtCore.QModelIndex()):

        if parent.column() > 0:

            return 0

        if not parent.isValid():

            p_item = self.rootItem

        else:

            p_item = parent.internalPointer()

        return p_item.childCount()




class LabeledCircleWidget(QtWidgets.QWidget):
        

    def __init__(self, title ='', factor = 1):

        super(LabeledCircleWidget, self).__init__()

        self._title = title

        self.title_label = QtWidgets.QLabel(self.title)

        self.value_label = QtWidgets.QLineEdit('')

        self.value_label.setValidator(QtGui.QDoubleValidator())

        self.value_label.returnPressed.connect(self.update_value)

        self.value_label.setFixedWidth(50)

        self.dial = QtWidgets.QDial()

        self.dial.valueChanged.connect(self.value_changed)

        self.dial.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self.value_label.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)

        self.v_layout = QtWidgets.QVBoxLayout()

        self.h_layout = QtWidgets.QHBoxLayout()

        self.h_layout.addWidget(self.dial)

        self.h_layout.addWidget(self.value_label)

        self.v_layout.addWidget(self.title_label)

        self.v_layout.addLayout(self.h_layout)

        self.setLayout(self.v_layout)

        self.valueChanged = self.dial.valueChanged

        self.setMinimum = self.dial.setMinimum

        self.setMaximum = self.dial.setMaximum

        self.setValue = self.dial.setValue

        self.value = self.dial.value

        self.setSingleStep = self.dial.setSingleStep

        self.setEnabled = self.dial.setEnabled

        self.setRange = self.dial.setRange

        self.factor = factor

        self.value_changed(self.dial.value())



    @property

    def title(self):

        return self._title



    @title.setter

    def title(self, value):

        self._title = value

        self.title_label.setText(value)


    def update_value(self):

        try:

            self.dial.setValue(int(float(self.value_label.text()) * self.factor))

        except ValueError:

            pass



    def value_changed(self, value: int) -> None:

        self.value_label.setText(str(value))






class LabeledSlider(QtWidgets.QWidget):

    def __init__(self, title='', factor=1) -> None:

        super(LabeledSlider, self).__init__()

        self._title = title

        self.title_label = QtWidgets.QLabel(self.title)

        self.value_label = QtWidgets.QLineEdit('')

        self.value_label.setValidator(QtGui.QDoubleValidator())

        self.value_label.returnPressed.connect(self.update_value)

        self.value_label.setFixedWidth(50)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)

        self.slider.valueChanged.connect(self.value_changed)

        self.slider.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self.value_label.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)

        self.v_layout = QtWidgets.QVBoxLayout()

        self.h_layout = QtWidgets.QHBoxLayout()

        self.h_layout.addWidget(self.slider)

        self.h_layout.addWidget(self.value_label)

        self.v_layout.addWidget(self.title_label)

        self.v_layout.addLayout(self.h_layout)

        self.setLayout(self.v_layout)

        self.valueChanged = self.slider.valueChanged

        self.setMinimum = self.slider.setMinimum

        self.setMaximum = self.slider.setMaximum

        self.setValue = self.slider.setValue

        self.value = self.slider.value

        self.setSingleStep = self.slider.setSingleStep

        self.setEnabled = self.slider.setEnabled

        self.setRange = self.slider.setRange

        self.factor = factor

        self.value_changed(self.slider.value())



    def update_value(self):

        try:

            self.slider.setValue(int(float(self.value_label.text())*self.factor))

        except ValueError:

            pass



    def value_changed(self, value: int) -> None:

        # self.slider.valueChanged(value)

        self.value_label.setText(str(value/self.factor))



    @property

    def title(self):

        return self._title



    @title.setter

    def title(self, value):

        self._title = value

        self.title_label.setText(value)





class EditViewBox(ViewBox):

    rotation = QtCore.pyqtSignal(float)

    translation = QtCore.pyqtSignal(float, float)

    scale = QtCore.pyqtSignal(float)

    cell_select = QtCore.pyqtSignal(float, float, float, float)


    def __init__(self, parent=None, border=None, lockAspect=False, enableMouse=True, invertY=False, enableMenu=False,

                 name=None, invertX=False):

        super().__init__(parent, border, lockAspect, enableMouse, invertY, enableMenu, name, invertX)

        self._logger = logging.getLogger('Atlaslog')



    def mouseClickEvent(self, ev):

        ev.accept()

        x = ev.pos().x()
        
        y = ev.pos().y()

        v_range = self.viewRange()

        self.cell_select.emit(x, y, v_range[0][0], v_range[1][0])

        super().mouseClickEvent(ev)



    def mouseDragEvent(self, ev, axis=None):

        ev.accept()

        x_shift = ev.lastPos().x() - ev.pos().x()

        y_shift = ev.lastPos().y() - ev.pos().y()

        d = sqrt(x_shift ** 2 + y_shift ** 2)

        sign = 1 if y_shift > 0 else -1

        d *= sign

        if ev.modifiers() == QtCore.Qt.ControlModifier:

            if ev.isFinish() or True:

                self.rotation.emit(d)

            return

        elif ev.modifiers() == QtCore.Qt.ShiftModifier:

            if ev.isFinish() or True:

                self.translation.emit(y_shift, x_shift)

            return

        elif ev.modifiers() == QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier:

            self.scale.emit(y_shift)

            return


        super().mouseDragEvent(ev, axis)





class Transform(QtCore.QObject):

    rotation_changed = QtCore.pyqtSignal(float)

    translation_changed = QtCore.pyqtSignal(float, float)

    scale_changed = QtCore.pyqtSignal(float)

    fliplr_changed = QtCore.pyqtSignal(bool)

    flipud_changed = QtCore.pyqtSignal(bool)



    def __init__(self, translation=(0, 0), rotation=0, scale=1., flipped_lr=False, flipped_ud=False) -> None:

        super(Transform, self).__init__()

        self._translation = translation

        self._rotation = rotation

        self._scale = scale

        self._flipped_lr = flipped_lr

        self._flipped_ud = flipped_ud



    @property

    def flipped_lr(self):

        return self._flipped_lr



    @flipped_lr.setter

    def flipped_lr(self, value):

        self._flipped_lr = value

        self.fliplr_changed.emit(value)



    @property

    def flipped_ud(self):

        return self._flipped_ud



    @flipped_ud.setter

    def flipped_ud(self, value):

        self._flipped_ud = value

        self.flipud_changed.emit(value)



    @property

    def rotation(self):

        return self._rotation



    @rotation.setter

    def rotation(self, value):

        self._rotation = value

        self.rotation_changed.emit(value)



    @property

    def translation(self):

        return self._translation



    @translation.setter

    def translation(self, value):

        self._translation = value

        self.translation_changed.emit(value[0], value[1])



    @property

    def scale(self):

        return self._scale



    @scale.setter

    def scale(self, value):

        self._scale = value

        self.scale_changed.emit(value)



    def add_rotation(self, angle):

        self.rotation += angle

        self.rotation_changed.emit(self.rotation)



    def add_translation(self, trans):

        self.translation = (self.translation[0] + trans[0], self.translation[1] + trans[1])

        self.translation_changed.emit(self.translation[0], self.translation[1])



    def add_scale(self, scale):

        self.scale += scale

        self.scale_changed.emit(self.scale)



    @property

    def params(self):

        d = {'x_shift': self.translation[0], 'y_shift': self.translation[1],

             'rotation': self.rotation, 'scale': self.scale,

             'flip_lr': self.flipped_lr, 'flip_ud': self.flipped_ud}

        return d





class SliceImage(QtCore.QObject):

    img_updated = QtCore.pyqtSignal()

    max_value_changed = QtCore.pyqtSignal(float)

    n_slices_known = QtCore.pyqtSignal(int, int, int, int)  # n_zslices, n_channels, n_res_levels, n_brainslices



    def __init__(self, path):

        super(SliceImage, self).__init__()

        self._logger = logging.getLogger('Atlaslog')

        self.path = Path(path)

        self.downfactor = 1

        self._raw_img = Image.new('L', (50, 50))

        self._dw_img = Image.new('L', (50, 50))

        self._img = Image.new('L', (50, 50))

        self.img = self._img

        self._img_vsi = None

        self._ndpis_files = ()

        self._bboxes = None

        self._is_tiff = None

        self._is_ndpis = None

        self.p_max = 2 ** 16

        self.real_scale = {'x_um': 1, 'y_um': 1, 'z_um': 1}

        self.channels = ()

        self._c_channel = 0

        self._c_zslice = 0

        self._c_slice = 0

        self._c_zoom = 4

        self._prms = {}

        self.stack_loader = self.default_stack_loader



    @property

    def dw_img(self):

        return self._dw_img



    @property

    def raw_img(self):

        return self._raw_img



    @property

    def is_ndpis(self):

        return self._is_ndpis



    @property

    def is_tiff(self):

        return self._is_tiff



    @property

    def c_zoom(self):

        return self._c_zoom



    @c_zoom.setter

    def c_zoom(self, value):

        self.stack_loader(z=self.c_zslice, res=value, channel=self.c_channel, brain_slice=self.c_slice)

        self._c_zoom = value



    @property

    def c_channel(self):

        # Different colors

        return self._c_channel



    @c_channel.setter

    def c_channel(self, value):

        # change color channel

        self.stack_loader(z=self.c_zslice, res=self.c_zoom, channel=value, brain_slice=self.c_slice)

        self._c_channel = value



    @property

    def c_slice(self):

        # In case of mutliple brain slices on the same slide

        return self._c_slice



    @c_slice.setter

    def c_slice(self, value):

        self.stack_loader(z=self.c_zslice, res=self.c_zoom, channel=self.c_channel, brain_slice=value)

        self._c_slice = value



    @property

    def c_zslice(self):

        return self._c_zslice



    @c_zslice.setter

    def c_zslice(self, value):

        self.stack_loader(z=value, res=self.c_zoom, channel=self.c_channel, brain_slice=self.c_slice)

        self._c_zslice = value



    @property

    def x_scale(self):

        return self.real_scale['x_um']



    @x_scale.setter

    def x_scale(self, value):

        self.real_scale['x_um'] = value



    @property

    def y_scale(self):

        return self.real_scale['y_um']



    @y_scale.setter

    def y_scale(self, value):

        self.real_scale['y_um'] = value



    @property

    def z_scale(self):

        return self.real_scale['z_um']



    @z_scale.setter

    def z_scale(self, value):

        self.real_scale['z_um'] = value



    @property

    def p_max(self):

        return self._p_max



    @p_max.setter

    def p_max(self, value):

        self._p_max = value

        self.max_value_changed.emit(value)



    @property

    def img(self):

        return self._img



    @img.setter

    def img(self, value):

        self._img = value

        self._pic = np.array(value)

        self.img_updated.emit()



    @property

    def pic(self):

        return self._pic



    def reset_img(self):

        self.img = self._dw_img.copy()



    def load(self):

        """

        Load the image defined in self.path

        Loading procedure depends on format. Handles tif through PIL, vsi through python-bioformats

        """

        if self.path is None:

            return

        self._is_tiff = False

        self._prms = {}

        self._c_zslice = 0

        self._c_zoom = 4

        self._c_channel = 0

        self._c_slice = 0

        if self.path.suffix in {'.tif', '.tiff'}:

            self.downfactor = 50

            self._raw_img = Image.open(self.path)

            self._dw_img = self._raw_img.resize((self._raw_img.width // self.downfactor,

                                                 self._raw_img.height // self.downfactor))

            self.img = self._dw_img.copy()

            self._is_tiff = True

        elif self.path.suffix in {'.vsi'}:

            self.load_slide_scanner_image()

            self.stack_loader = self.vsi_stack_loader

        elif self.path.suffix in {'.ndpis'}:

            self.stack_loader = self.ndpis_stack_loader

            self._is_ndpis = True

            self.ndpis_stack_loader(0, 4, 0, 0)

        self.p_max = self.pic.max()





    # Images of different format are handled through different functions whose signature is identical

    # It should be loader(self, z, res, channel, brain_slice) to be able to specify:

    # the z, the resolution, the channel and the brain slice for formats that handle multiple slice per file

    # The specific stack loader is then assigned to self.stack_loader

    # The rest of the image handling should be ~transparent



    # Slide scanner vsi files

    def slice_slide_scanner(self, c=0):

        pic = self._img_vsi[..., c]

        self._raw_img = Image.fromarray(pic)

        self._dw_img = self._raw_img.copy()

        self.img = self._dw_img.copy()



    def load_slide_scanner_image(self, z=0, serie=4):

        self.downfactor = 1

        self._img_vsi = bf.load_image(self.path.as_posix(), t=0, series=serie, z=z)

        metadata = bf.get_omexml_metadata(self.path.as_posix())

        o = bf.OMEXML(metadata)

        n_zslices = o.image_count

        self.n_slices_known.emit(n_zslices, self._img_vsi.shape[-1], 4, 1)

        self.slice_slide_scanner()



    def vsi_stack_loader(self, z=0, res=4, channel=0, brain_slice=None):

        if z != self.c_zslice or res != self.c_zoom:

            self.load_slide_scanner_image(z, res)

        if channel != self.c_channel:

            self.slice_slide_scanner(channel)



    # Nanozoomer ndpis files

    def ndpis_stack_loader(self, z, res, channel, brain_slice):

        """

        Stack loader for ndpis file



        Parameters

        ----------

        z

        res

        channel

        brain_slice



        Returns

        -------



        """

        if self._prms == {}:

            self._prms = nz.parse_ndpis(self.path)

            images = [self._prms[f'Image{ix}'] for ix in range(int(self._prms['NoImages']))]

            self._ndpis_files = [self._prms['path'] / im for im in images]

            # FIXME: If no image contains dapi this will fail

            dapi_path = [p for p, n in zip(self._ndpis_files, images) if 'dapi' in n.lower()][0]

            dapi_im = nz.open_im(dapi_path)

            self._bboxes = nz.get_bboxes(dapi_im)

            self.n_slices_known.emit(1, len(images), dapi_im.level_count, self._bboxes.shape[0])

            dapi_im.close()

            self._logger.debug(f'{self._bboxes}')

        parent_path = self.path.parent

        img_name = self.path.stem

        base_path = parent_path / img_name

        if not base_path.is_dir():

            base_path.mkdir()

        channel_path = base_path / f'ch{channel}'

        if not channel_path.is_dir():

            channel_path.mkdir()

        res_path = channel_path / f'res_{res}'

        if not res_path.is_dir():

            res_path.mkdir()

        img_path = res_path / f'{img_name}-{brain_slice}.tiff'

        if img_path.is_file():

            # Load existing file

            self._raw_img = Image.open(img_path)

        else:

            # Create and save

            im = nz.open_im(self._ndpis_files[channel])

            raw_crop = nz.crop(im, self._bboxes[brain_slice, :], res)

            self._raw_img = raw_crop

            raw_crop.save(img_path)

            im.close()

        self._dw_img = self._raw_img.copy()

        self.img = self._dw_img.copy()

        # Load new image



    def default_stack_loader(self, z, res, channel, brain_slice=None):

        pass


