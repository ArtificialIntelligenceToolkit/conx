# conx - a neural network library
#
# Copyright (c) Douglas S. Blank <doug.blank@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA 02110-1301  USA

import numpy as np
import threading
import time

from IPython.display import Javascript, display
import ipywidgets
from ipywidgets import (HTML, Button, VBox, HBox, IntSlider, Select, Text,
                        Layout, Label, FloatSlider, Checkbox, IntText,
                        Box, Accordion, FloatText, Output, Widget, register,
                        widget_serialization, DOMWidget)
from traitlets import Bool, Dict, Int, Float, Unicode, List, Instance

from .utils import uri_to_image, AVAILABLE_COLORMAPS, get_colormap
from ._version import __version__

class _Player(threading.Thread):
    """
    Background thread for running a player.
    """
    def __init__(self, controller, time_wait=0.5):
        self.controller = controller
        threading.Thread.__init__(self)
        self.time_wait = time_wait
        self.can_run = threading.Event()
        self.can_run.clear()  ## paused
        self.daemon =True ## allows program to exit without waiting for join

    def run(self):
        while True:
            self.can_run.wait()
            self.controller.goto("next")
            time.sleep(self.time_wait)

    def pause(self):
        self.can_run.clear()

    def resume(self):
        self.can_run.set()

class SequenceViewer(VBox):
    def __init__(self, title, function, length, play_rate=0.5):
        self.player = _Player(self, play_rate)
        self.player.start()
        self.title = title
        self.function = function
        self.length = length
        self.output = Output()
        self.position_text = IntText(value=0, layout=Layout(width="100%"))
        self.total_text = Label(value="of %s" % self.length, layout=Layout(width="100px"))
        controls = self.make_controls()
        super().__init__([controls, self.output])

    def goto(self, position):
        #### Position it:
        if position == "begin":
            self.control_slider.value = 0
        elif position == "end":
            self.control_slider.value = self.length - 1
        elif position == "prev":
            if self.control_slider.value - 1 < 0:
                self.control_slider.value = self.length - 1 # wrap around
            else:
                self.control_slider.value = max(self.control_slider.value - 1, 0)
        elif position == "next":
            if self.control_slider.value + 1 > self.length - 1:
                self.control_slider.value = 0 # wrap around
            else:
                self.control_slider.value = min(self.control_slider.value + 1, self.length - 1)
        self.position_text.value = self.control_slider.value

    def toggle_play(self, button):
        ## toggle
        if self.button_play.description == "Play":
            self.button_play.description = "Stop"
            self.button_play.icon = "pause"
            self.player.resume()
        else:
            self.button_play.description = "Play"
            self.button_play.icon = "play"
            self.player.pause()

    def make_controls(self):
        button_begin = Button(icon="fast-backward", layout=Layout(width='100%'))
        button_prev = Button(icon="backward", layout=Layout(width='100%'))
        button_next = Button(icon="forward", layout=Layout(width='100%'))
        button_end = Button(icon="fast-forward", layout=Layout(width='100%'))
        self.button_play = Button(icon="play", description="Play", layout=Layout(width="100%"))
        self.control_buttons = HBox([
            button_begin,
            button_prev,
            self.position_text,
            button_next,
            button_end,
            self.button_play,
        ], layout=Layout(width='100%', height="50px"))
        self.control_slider = IntSlider(description=self.title,
                                        continuous_update=False,
                                        min=0,
                                        max=max(self.length - 1, 0),
                                        value=0,
                                        style={"description_width": 'initial'},
                                        layout=Layout(width='100%'))
        ## Hook them up:
        button_begin.on_click(lambda button: self.goto("begin"))
        button_end.on_click(lambda button: self.goto("end"))
        button_next.on_click(lambda button: self.goto("next"))
        button_prev.on_click(lambda button: self.goto("prev"))
        self.button_play.on_click(self.toggle_play)
        self.control_slider.observe(self.update_slider_control, names='value')
        controls = VBox([HBox([self.control_slider, self.total_text], layout=Layout(height="40px")),
                         self.control_buttons], layout=Layout(width='100%'))
        controls.on_displayed(lambda widget: self.initialize())
        return controls

    def initialize(self):
        results = self.function(self.control_slider.value)
        try:
            results = list(results)
        except:
            results = [results]
        self.displayers = [display(x, display_id=True) for x in results]

    def update_slider_control(self, change):
        if change["name"] == "value":
            self.position_text.value = self.control_slider.value
            self.output.clear_output(wait=True)
            results = self.function(self.control_slider.value)
            try:
                results = list(results)
            except:
                results = [results]
            for i in range(len(self.displayers)):
                self.displayers[i].update(results[i])

class Dashboard(VBox):
    """
    Build the dashboard for Jupyter widgets. Requires running
    in a notebook/jupyterlab.
    """
    def __init__(self, net, width="95%", height="550px", play_rate=0.5):
        self._ignore_layer_updates = False
        self.player = _Player(self, play_rate)
        self.player.start()
        self.net = net
        self._width = width
        self._height = height
        ## Global widgets:
        style = {"description_width": "initial"}
        self.feature_columns = IntText(description="Feature columns:", value=3, style=style)
        self.feature_scale = FloatText(description="Feature scale:", value=2.0, style=style)
        self.feature_columns.observe(self.regenerate, names='value')
        self.feature_scale.observe(self.regenerate, names='value')
        ## Hack to center SVG as justify-content is broken:
        self.net_svg = HTML(value="""<p style="text-align:center">%s</p>""" % ("",), layout=Layout(
            width=self._width, overflow_x='auto', overflow_y="auto",
            justify_content="center"))
        # Make controls first:
        self.output = Output()
        controls = self.make_controls()
        config = self.make_config()
        super().__init__([config, controls, self.net_svg, self.output])

    def goto(self, position):
        if len(self.net.dataset.inputs) == 0 or len(self.net.dataset.targets) == 0:
            return
        if self.control_select.value == "Train":
            length = len(self.net.dataset.train_inputs)
        elif self.control_select.value == "Test":
            length = len(self.net.dataset.test_inputs)
        #### Position it:
        if position == "begin":
            self.control_slider.value = 0
        elif position == "end":
            self.control_slider.value = length - 1
        elif position == "prev":
            if self.control_slider.value - 1 < 0:
                self.control_slider.value = length - 1 # wrap around
            else:
                self.control_slider.value = max(self.control_slider.value - 1, 0)
        elif position == "next":
            if self.control_slider.value + 1 > length - 1:
                self.control_slider.value = 0 # wrap around
            else:
                self.control_slider.value = min(self.control_slider.value + 1, length - 1)
        self.position_text.value = self.control_slider.value

    def update_control_slider(self, change=None):
        if len(self.net.dataset.inputs) == 0 or len(self.net.dataset.targets) == 0:
            self.total_text.value = "of 0"
            self.control_slider.value = 0
            self.position_text.value = 0
            self.control_slider.disabled = True
            self.position_text.disabled = True
            for child in self.control_buttons.children:
                if not hasattr(child, "icon") or child.icon != "refresh":
                    child.disabled = True
            return
        if self.control_select.value == "Test":
            self.total_text.value = "of %s" % len(self.net.dataset.test_inputs)
            minmax = (0, max(len(self.net.dataset.test_inputs) - 1, 0))
            if minmax[0] <= self.control_slider.value <= minmax[1]:
                pass # ok
            else:
                self.control_slider.value = 0
            self.control_slider.min = minmax[0]
            self.control_slider.max = minmax[1]
            if len(self.net.dataset.test_inputs) == 0:
                disabled = True
            else:
                disabled = False
        elif self.control_select.value == "Train":
            self.total_text.value = "of %s" % len(self.net.dataset.train_inputs)
            minmax = (0, max(len(self.net.dataset.train_inputs) - 1, 0))
            if minmax[0] <= self.control_slider.value <= minmax[1]:
                pass # ok
            else:
                self.control_slider.value = 0
            self.control_slider.min = minmax[0]
            self.control_slider.max = minmax[1]
            if len(self.net.dataset.train_inputs) == 0:
                disabled = True
            else:
                disabled = False
        self.control_slider.disabled = disabled
        self.position_text.disbaled = disabled
        self.position_text.value = self.control_slider.value
        for child in self.control_buttons.children:
            if not hasattr(child, "icon") or child.icon != "refresh":
                child.disabled = disabled

    def update_zoom_slider(self, change):
        if change["name"] == "value":
            self.net.config["svg_height"] = self.zoom_slider.value * 780
            self.regenerate()

    def update_position_text(self, change):
        if (change["name"] == "_property_lock" and
            isinstance(change["new"], dict) and
            "value" in change["new"]):
            self.control_slider.value = change["new"]["value"]

    def get_current_input(self):
        if self.control_select.value == "Train" and len(self.net.dataset.train_targets) > 0:
            return self.net.dataset.train_inputs[self.control_slider.value]
        elif self.control_select.value == "Test" and len(self.net.dataset.test_targets) > 0:
            return self.net.dataset.test_inputs[self.control_slider.value]

    def update_slider_control(self, change):
        if len(self.net.dataset.inputs) == 0 or len(self.net.dataset.targets) == 0:
            self.total_text.value = "of 0"
            return
        if change["name"] == "value":
            self.position_text.value = self.control_slider.value
            if self.control_select.value == "Train" and len(self.net.dataset.train_targets) > 0:
                self.total_text.value = "of %s" % len(self.net.dataset.train_inputs)
                output = self.net.propagate(self.net.dataset.train_inputs[self.control_slider.value], visualize=True)
                if self.feature_bank.value in self.net.layer_dict.keys():
                    self.net.propagate_to_features(self.feature_bank.value, self.net.dataset.train_inputs[self.control_slider.value],
                                               cols=self.feature_columns.value, scale=self.feature_scale.value, html=False)
                if self.net.config["show_targets"]:
                    if len(self.net.output_bank_order) == 1:
                        self.net.display_component([self.net.dataset.train_targets[self.control_slider.value]], "targets", minmax=(-1, 1))
                    else:
                        self.net.display_component(self.net.dataset.train_targets[self.control_slider.value], "targets", minmax=(-1, 1))
                if self.net.config["show_errors"]:
                    if len(self.net.output_bank_order) == 1:
                        errors = np.array(output) - np.array(self.net.dataset.train_targets[self.control_slider.value])
                        self.net.display_component([errors.tolist()], "errors", minmax=(-1, 1))
                    else:
                        errors = []
                        for bank in range(len(self.net.output_bank_order)):
                            errors.append( np.array(output[bank]) - np.array(self.net.dataset.train_targets[self.control_slider.value][bank]))
                        self.net.display_component(errors, "errors", minmax=(-1, 1))
            elif self.control_select.value == "Test" and len(self.net.dataset.test_targets) > 0:
                self.total_text.value = "of %s" % len(self.net.dataset.test_inputs)
                output = self.net.propagate(self.net.dataset.test_inputs[self.control_slider.value], visualize=True)
                if self.feature_bank.value in self.net.layer_dict.keys():
                    self.net.propagate_to_features(self.feature_bank.value, self.net.dataset.test_inputs[self.control_slider.value],
                                               cols=self.feature_columns.value, scale=self.feature_scale.value, html=False)
                if self.net.config["show_targets"]:
                    self.net.display_component([self.net.dataset.test_targets[self.control_slider.value]], "targets", minmax=(-1, 1))
                if self.net.config["show_errors"]:
                    if len(self.net.output_bank_order) == 1:
                        errors = np.array(output) - np.array(self.net.dataset.test_targets[self.control_slider.value])
                        self.net.display_component([errors.tolist()], "errors", minmax=(-1, 1))
                    else:
                        errors = []
                        for bank in range(len(self.net.output_bank_order)):
                            errors.append( np.array(output[bank]) - np.array(self.net.dataset.test_targets[self.control_slider.value][bank]))
                        self.net.display_component(errors, "errors", minmax=(-1, 1))

    def train_one(self, button):
        if len(self.net.dataset.inputs) == 0 or len(self.net.dataset.targets) == 0:
            return
        if self.control_select.value == "Train" and len(self.net.dataset.train_targets) > 0:
            outputs = self.train_one(self.net.dataset.train_inputs[self.control_slider.value],
                                     self.net.dataset.train_targets[self.control_slider.value])
        elif self.control_select.value == "Test" and len(self.net.dataset.test_targets) > 0:
            outputs = self.train_one(self.net.dataset.test_inputs[self.control_slider.value],
                                     self.net.dataset.test_targets[self.control_slider.value])

    def toggle_play(self, button):
        ## toggle
        if self.button_play.description == "Play":
            self.button_play.description = "Stop"
            self.button_play.icon = "pause"
            self.player.resume()
        else:
            self.button_play.description = "Play"
            self.button_play.icon = "play"
            self.player.pause()

    def prop_one(self, button=None):
        self.update_slider_control({"name": "value"})

    def regenerate(self, button=None):
        ## Protection when deleting object on shutdown:
        if isinstance(button, dict) and 'new' in button and button['new'] is None:
            return
        #with self.output:
        #    print("regenerate called!", button)
        inputs = self.get_current_input()
        features = None
        if self.feature_bank.value in self.net.layer_dict.keys():
            features = self.net.propagate_to_features(self.feature_bank.value, inputs,
                                                      cols=self.feature_columns.value,
                                                      scale=self.feature_scale.value, display=False)
        svg = """<p style="text-align:center">%s</p>""" % (self.net.build_svg(inputs=inputs),)
        if inputs is not None and features is not None:
            self.net_svg.value = """
<table align="center" style="width: 100%%;">
 <tr>
  <td valign="top">%s</td>
  <td valign="top" align="center"><p style="text-align:center"><b>%s</b></p>%s</td>
</tr>
</table>""" % (svg, "%s features" % self.feature_bank.value, features)
        else:
            self.net_svg.value = svg

    def make_colormap_image(self, colormap_name):
        from .layers import Layer
        if not colormap_name:
            colormap_name = get_colormap()
        layer = Layer("Colormap", 100)
        image = layer.make_image(np.arange(-1, 1, .01), colormap_name,
                                 {"pixels_per_unit": 1}).resize((300, 25))
        return image

    def set_attr(self, obj, attr, value):
        if value not in [{}, None]: ## value is None when shutting down
            if isinstance(value, dict):
                value = value["value"]
            if isinstance(obj, dict):
                obj[attr] = value
            else:
                setattr(obj, attr, value)
            ## was crashing on Widgets.__del__, if get_ipython() no longer existed
            self.regenerate()

    def make_controls(self):
        button_begin = Button(icon="fast-backward", layout=Layout(width='100%'))
        button_prev = Button(icon="backward", layout=Layout(width='100%'))
        button_next = Button(icon="forward", layout=Layout(width='100%'))
        button_end = Button(icon="fast-forward", layout=Layout(width='100%'))
        #button_prop = Button(description="Propagate", layout=Layout(width='100%'))
        #button_train = Button(description="Train", layout=Layout(width='100%'))
        self.button_play = Button(icon="play", description="Play", layout=Layout(width="100%"))
        refresh_button = Button(icon="refresh", layout=Layout(width="25%"))

        self.position_text = IntText(value=0, layout=Layout(width="100%"))

        self.control_buttons = HBox([
            button_begin,
            button_prev,
            #button_train,
            self.position_text,
            button_next,
            button_end,
            self.button_play,
            refresh_button
        ], layout=Layout(width='100%', height="50px"))
        length = (len(self.net.dataset.train_inputs) - 1) if len(self.net.dataset.train_inputs) > 0 else 0
        self.control_slider = IntSlider(description="Dataset index",
                                   continuous_update=False,
                                   min=0,
                                   max=max(length, 0),
                                   value=0,
                                   layout=Layout(width='100%'))
        self.total_text = Label(value="of %s" % len(self.net.dataset.train_inputs), layout=Layout(width="100px"))
        self.zoom_slider = FloatSlider(description="Zoom", continuous_update=False, min=.5, max=3,
                                  value=self.net.config["svg_height"]/780.0)

        ## Hook them up:
        button_begin.on_click(lambda button: self.goto("begin"))
        button_end.on_click(lambda button: self.goto("end"))
        button_next.on_click(lambda button: self.goto("next"))
        button_prev.on_click(lambda button: self.goto("prev"))
        self.button_play.on_click(self.toggle_play)
        self.control_slider.observe(self.update_slider_control, names='value')
        refresh_button.on_click(lambda widget: (self.update_control_slider(),
                                                self.output.clear_output(),
                                                self.regenerate()))
        self.zoom_slider.observe(self.update_zoom_slider, names='value')
        self.position_text.observe(self.update_position_text, names='value')
        # Put them together:
        controls = VBox([HBox([self.control_slider, self.total_text], layout=Layout(height="40px")),
                         self.control_buttons], layout=Layout(width='100%'))

        #net_page = VBox([control, self.net_svg], layout=Layout(width='95%'))
        controls.on_displayed(lambda widget: self.regenerate())
        return controls

    def make_config(self):
        layout = Layout()
        style = {"description_width": "initial"}
        checkbox1 = Checkbox(description="Show Targets", value=self.net.config["show_targets"],
                             layout=layout, style=style)
        checkbox1.observe(lambda change: self.set_attr(self.net.config, "show_targets", change["new"]), names='value')
        checkbox2 = Checkbox(description="Errors", value=self.net.config["show_errors"],
                             layout=layout, style=style)
        checkbox2.observe(lambda change: self.set_attr(self.net.config, "show_errors", change["new"]), names='value')

        hspace = IntText(value=self.net.config["hspace"], description="Horizontal space between banks:",
                         style=style, layout=layout)
        hspace.observe(lambda change: self.set_attr(self.net.config, "hspace", change["new"]), names='value')
        vspace = IntText(value=self.net.config["vspace"], description="Vertical space between layers:",
                         style=style, layout=layout)
        vspace.observe(lambda change: self.set_attr(self.net.config, "vspace", change["new"]), names='value')
        self.feature_bank = Select(description="Features:", value="",
                              options=[""] + [layer.name for layer in self.net.layers if self.net._layer_has_features(layer.name)],
                              rows=1)
        self.feature_bank.observe(self.regenerate, names='value')
        self.control_select = Select(
            options=['Test', 'Train'],
            value='Train',
            description='Dataset:',
            rows=1
        )
        self.control_select.observe(self.update_control_slider, names='value')
        column1 = [self.control_select,
                   self.zoom_slider,
                   hspace,
                   vspace,
                   HBox([checkbox1, checkbox2]),
                   self.feature_bank,
                   self.feature_columns,
                   self.feature_scale
        ]
        ## Make layer selectable, and update-able:
        column2 = []
        layer = self.net.layers[-1]
        self.layer_select = Select(description="Layer:", value=layer.name,
                                   options=[layer.name for layer in
                                            self.net.layers],
                                   rows=1)
        self.layer_select.observe(self.update_layer_selection, names='value')
        column2.append(self.layer_select)
        self.layer_visible_checkbox = Checkbox(description="Visible", value=layer.visible, layout=layout)
        self.layer_visible_checkbox.observe(self.update_layer, names='value')
        column2.append(self.layer_visible_checkbox)
        self.layer_colormap = Select(description="Colormap:",
                                     options=[""] + AVAILABLE_COLORMAPS,
                                     value=layer.colormap if layer.colormap is not None else "", layout=layout, rows=1)
        self.layer_colormap_image = HTML(value="""<img src="%s"/>""" % self.net._image_to_uri(self.make_colormap_image(layer.colormap)))
        self.layer_colormap.observe(self.update_layer, names='value')
        column2.append(self.layer_colormap)
        column2.append(self.layer_colormap_image)
        self.layer_mindim = FloatText(description="Leftmost color maps to:", value=layer.minmax[0], style=style)
        self.layer_maxdim = FloatText(description="Rightmost color maps to:", value=layer.minmax[1], style=style)
        self.layer_mindim.observe(self.update_layer, names='value')
        self.layer_maxdim.observe(self.update_layer, names='value')
        column2.append(self.layer_mindim)
        column2.append(self.layer_maxdim)
        output_shape = layer.get_output_shape()
        self.layer_feature = IntText(value=layer.feature, description="Feature to show:", style=style)
        self.layer_feature.observe(self.update_layer, names='value')
        column2.append(self.layer_feature)

        config_children = HBox([VBox(column1, layout=Layout(width="100%")),
                                VBox(column2, layout=Layout(width="100%"))])
        accordion = Accordion(children=[config_children])
        accordion.set_title(0, self.net.name)
        accordion.selected_index = None
        return accordion

    def update_layer(self, change):
        """
        Update the layer object, and redisplay.
        """
        if self._ignore_layer_updates:
            return
        ## The rest indicates a change to a display variable.
        ## We need to save the value in the layer, and regenerate
        ## the display.
        # Get the layer:
        layer = self.net[self.layer_select.value]
        # Save the changed value in the layer:
        layer.feature = self.layer_feature.value
        layer.visible = self.layer_visible_checkbox.value
        ## These three, dealing with colors of activations,
        ## can be done with a prop_one():
        if "color" in change["owner"].description.lower():
            ## Matches: Colormap, lefmost color, rightmost color
            layer.minmax = (self.layer_mindim.value, self.layer_maxdim.value)
            layer.minmax = (self.layer_mindim.value, self.layer_maxdim.value)
            layer.colormap = self.layer_colormap.value if self.layer_colormap.value else None
            self.layer_colormap_image.value = """<img src="%s"/>""" % self.net._image_to_uri(self.make_colormap_image(layer.colormap))
            self.prop_one()
        else:
            self.regenerate()

    def update_layer_selection(self, change):
        """
        Just update the widgets; don't redraw anything.
        """
        ## No need to redisplay anything
        self._ignore_layer_updates = True
        ## First, get the new layer selected:
        layer = self.net[self.layer_select.value]
        ## Now, let's update all of the values without updating:
        self.layer_visible_checkbox.value = layer.visible
        self.layer_colormap.value = layer.colormap if layer.colormap != "" else ""
        self.layer_colormap_image.value = """<img src="%s"/>""" % self.net._image_to_uri(self.make_colormap_image(layer.colormap))
        self.layer_mindim.value = layer.minmax[0]
        self.layer_maxdim.value = layer.minmax[1]
        self.layer_feature.value = layer.feature
        self._ignore_layer_updates = False

@register("CameraWidget")
class CameraWidget(DOMWidget):
    """Represents a media source."""
    _view_module = Unicode('camera').tag(sync=True)
    _view_name = Unicode('CameraView').tag(sync=True)
    _model_module = Unicode('camera').tag(sync=True)
    _model_name = Unicode('CameraModel').tag(sync=True)
    _view_module_version = Unicode(__version__).tag(sync=True)
    # Specify audio constraint and video constraint as a boolean or dict.
    audio = Bool(False).tag(sync=True)
    video = Bool(True).tag(sync=True)
    image = Unicode('').tag(sync=True)
    image_count = Int(0).tag(sync=True)

    def __init__(self, *args, **kwargs):
        display(Javascript(get_camera_javascript()))
        super().__init__(*args, **kwargs)

    def get_image(self):
        if self.image:
            image = uri_to_image(self.image)
            image = image.convert("RGB")
            return image

    def get_data(self):
        if self.image:
            image = uri_to_image(self.image)
            ## trim from 4 to 3 dimensions: (remove alpha)
            # remove the 3 index of dimension index 2 (the A of RGBA color)
            image = np.delete(image, np.s_[3], 2)
            return (np.array(image).astype("float32") / 255.0)

def get_camera_javascript(width=320, height=240):
    if ipywidgets._version.version_info < (7,):
        jupyter_widgets = "jupyter-js-widgets"
    else:
        jupyter_widgets = "@jupyter-widgets/base"
    camera_javascript = """
require.undef('camera');

define('camera', ["%(jupyter_widgets)s"], function(widgets) {
    var CameraView = widgets.DOMWidgetView.extend({
        defaults: _.extend({}, widgets.DOMWidgetView.prototype.defaults, {
            _view_name: 'CameraView',
            _view_module: 'camera',
            audio: false,
            video: true,
        }),

        initialize: function() {

            var div = document.createElement('div');
            var el = document.createElement('video');
            el.setAttribute('id', "video_widget");
            el.setAttribute('width', %(width)s);
            el.setAttribute('height', %(height)s);
            div.appendChild(el);
            var canvas = document.createElement('canvas');
            canvas.setAttribute('id', "video_canvas");
            canvas.setAttribute('width', %(width)s);
            canvas.setAttribute('height', %(height)s);
            div.appendChild(canvas);
            div.appendChild(document.createElement('br'));
            var button = document.createElement('button');
            button.innerHTML = "Take a Picture";
            var that = this;
            button.onclick = function(b) {
                var video = document.querySelector("#video_widget");
                var canvas = document.querySelector("#video_canvas");
                if (video) {
                    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
                    var url = canvas.toDataURL('image/png');
                    if (that.model) {
                        that.model.set('image', url);
                        that.model.save_changes();
                    }
                }
            };
            div.appendChild(button);
            this.setElement(div);
            CameraView.__super__.initialize.apply(this, arguments);
        },

        render: function() {
            var that = this;
             that.model.stream.then(function(stream) {
                 that.el.children[0].src = window.URL.createObjectURL(stream);
                 that.el.children[0].play();
             });
        }
    });

    var CameraModel = widgets.DOMWidgetModel.extend({
        defaults: _.extend({}, widgets.DOMWidgetModel.prototype.defaults, {
            _model_name: 'CameraModel',
            _model_module: 'camera',
            audio: false,
            video: true
        }),

        initialize: function() {
            CameraModel.__super__.initialize.apply(this, arguments);
            // Get the camera permissions
            this.stream = navigator.mediaDevices.getUserMedia({audio: false, video: true});
        }
    });
    return {
        CameraModel: CameraModel,
        CameraView: CameraView
    }
});
""" % {"width": width, "height": height, "jupyter_widgets": jupyter_widgets}
    return camera_javascript
