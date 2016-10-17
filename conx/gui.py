from __future__ import print_function, division

from calysto.graphics import Canvas, Rectangle, Color

# def showNetwork(self, title=None, style='arrows', horizSpacing=40, vertSpacing=40, invert=False, **kw_args):
#     if title is None:
#         title = self.name
#     info = self.process_keyword_args(kw_args)
#     if self.actDisplay is not None and not self.actDisplay.closed:
#         self.actDisplay.close()
#     self.actDisplay = NetworkActivationDisplay(self, title, style, horizSpacing, vertSpacing, invert, info)
#     self.updateGraphics()


class VisualNetwork():
    def __init__(self, net, title, style, horizSpacing, vertSpacing, invert, info):
        validStyles = ['line', 'lines', 'arrow', 'arrows', 'full', 'fullarrows']
        assert style in validStyles, 'style must be one of: %s' % ', '.join(validStyles)
        defaultScale = 30
        sideMargin = 40
        topMargin = 30
        bottomMargin = 20
        textSize = 14
        self.bgColor = Color(176, 196, 222)  # 'light steel blue'
        fgText = 'blue'
        bgText = self.bgColor
        self.wrongColor = 'red'
        self.wrong = False
        self.invert = invert
        self.net = net
        self.info = info
        self.rectangles = {}
        for name in info:
            # fill in any shape or scale info not provided by the user
            if 'shape' not in info[name]:
                info[name]['shape'] = 1, net[name].size
            if 'scale' not in info[name]:
                info[name]['scale'] = defaultScale
            rows, cols = info[name]['shape']
            assert rows*cols == net[name].size, 'mismatched shape for layer %s' % name
            scale = info[name]['scale']
            info[name]['width'] = cols * scale
            info[name]['height'] = rows * scale
            info[name]['size'] = net[name].size
        # assign level numbers to each layer
        levelNums = []
        for layer in net.layers:
            if layer.kind == 'Undefined':
                raise Exception('bad network architecture')
            elif layer.kind in ('Input', 'Context'):
                info[layer.name]['level'] = 1
            elif 'level' not in info[layer.name]:
                prevLayers = net.getIncomingLayers(layer.name)
                prevLevels = [info[l.name]['level'] for l in prevLayers]
                info[layer.name]['level'] = max(prevLevels) + 1
            levelNums.append((layer.name, info[layer.name]['level']))
#        print 'levelNums:', levelNums
        maxLevel = max([num for (name, num) in levelNums])
        # group layers together by level
        # example: levels = [['input', 'context'] ['hidden'] ['output']]
        levels = []
        for i in range(1, maxLevel+1):
            levels.append([name for (name, num) in levelNums if num == i])
#        print 'levels:', levels
        # figure out window size
        widths, heights = [], []
        for level in levels:
            if level == []:
                raise Exception('bad network architecture')
            levelWidth = sum([info[name]['width'] for name in level]) + (len(level)-1)*horizSpacing
            widths.append(levelWidth)
            levelHeight = max([info[name]['height'] for name in level]) + textSize
            heights.append(levelHeight)
        width = max(widths) + 2*sideMargin
        height = sum(heights) + (len(levels)-1)*vertSpacing + topMargin + bottomMargin
        # create window
        #GraphWin.__init__(self, title, width, height)
        self.setBackground(self.bgColor)
        labels = []
        centerX = width / 2
        baseline = height - bottomMargin
        # figure out bounding box of each layer at each level
        for level in levels:
            levelWidth = sum([info[name]['width'] for name in level]) + (len(level)-1)*horizSpacing
            maxHeight = max([info[name]['height'] for name in level])
            leftX = centerX - levelWidth / 2
            centerY = baseline - maxHeight / 2
            x1 = leftX
            for name in level:
                rows, cols = info[name]['shape']
                scale = info[name]['scale']
                w, h = info[name]['width'], info[name]['height']
                x2 = x1 + w
                y1 = centerY - h/2
                y2 = centerY + h/2
                info[name]['bbox'] = x1, y1, x2, y2
                # create activation rectangles
                rects = []
                x, y = x1, y1
                for r in xrange(rows):
                    for c in xrange(cols):
                        rect = self.create_rectangle(x, y, x+scale, y+scale, outline='black', fill='gray50')
                        rects.append(rect)
                        x += scale
                    x, y = x1, y+scale
                self.rectangles[name] = rects
                # create label
                textX = (x1 + x2) / 2
                textY = y1 - 0.8 * textSize
                if info[name]['size'] > 1:
                    label = Text(Point(textX, textY), '%s units' % name)
                else:
                    label = Text(Point(textX, textY), '%s unit' % name)
                label.setSize(textSize)
                labels.append(label)
                x1 = x2 + horizSpacing
            baseline = baseline - maxHeight - textSize - vertSpacing
        # draw feedforward connections
        lines = []
        for c in net.connections:
            from_x1, from_y1, from_x2, from_y2 = info[c.fromLayer.name]['bbox']
            to_x1, to_y1, to_x2, to_y2 = info[c.toLayer.name]['bbox']
            from_rows, from_cols = info[c.fromLayer.name]['shape']
            to_rows, to_cols = info[c.toLayer.name]['shape']
            from_scale = info[c.fromLayer.name]['scale']
            to_scale = info[c.toLayer.name]['scale']
            if style == 'line':
                from_xc = (from_x1 + from_x2) / 2
                to_xc = (to_x1 + to_x2) / 2
                l = Line(Point(from_xc, from_y1), Point(to_xc, to_y2))
                lines.append(l)
            elif style == 'lines':
                l1 = Line(Point(from_x1, from_y1), Point(to_x1, to_y2))
                l2 = Line(Point(from_x2, from_y1), Point(to_x2, to_y2))
                lines.extend([l1, l2])
            elif style == 'arrow':
                from_xc = (from_x1 + from_x2) / 2
                to_xc = (to_x1 + to_x2) / 2
                l = Line(Point(from_xc, from_y1), Point(to_xc, to_y2))
                l.setArrow('last')
                lines.append(l)
            elif style == 'arrows':
                l1 = Line(Point(from_x1, from_y1), Point(to_x1, to_y2))
                l1.setArrow('last')
                l2 = Line(Point(from_x2, from_y1), Point(to_x2, to_y2))
                l2.setArrow('last')
                lines.extend([l1, l2])
            elif style == 'full':
                from_x = from_x1 + from_scale/2
                for j in range(from_cols):
                    to_x = to_x1 + to_scale/2
                    for i in range(to_cols):
                        l = Line(Point(from_x, from_y1), Point(to_x, to_y2))
                        lines.append(l)
                        to_x += to_scale
                    from_x += from_scale
            elif style == 'fullarrows':
                from_x = from_x1 + from_scale/2
                for j in range(from_cols):
                    to_x = to_x1 + to_scale/2
                    for i in range(to_cols):
                        l = Line(Point(from_x, from_y1), Point(to_x, to_y2))
                        l.setArrow('last')
                        lines.append(l)
                        to_x += to_scale
                    from_x += from_scale
        # draw recurrent context connections if any
        if 'contextLayers' in dir(net):
            recurrent_width = 2
            for name in net.contextLayers:
                h_name = name
                c_name = net.contextLayers[h_name].name
                h_x1, h_y1, h_x2, h_y2 = info[h_name]['bbox']
                c_x1, c_y1, c_x2, c_y2 = info[c_name]['bbox']
                rec_y1 = (h_y1 + h_y2) / 2
                rec_y2 = (c_y1 + c_y2) / 2
                rec_x2 = max(h_x2, c_x2) + horizSpacing/2
                if rec_x2 >= width:
                    rec_x2 = (max(h_x2, c_x2) + width) / 2
                l1 = Line(Point(h_x2, rec_y1), Point(rec_x2, rec_y1))
                l1.setWidth(recurrent_width)
                l2 = Line(Point(rec_x2, rec_y1), Point(rec_x2, rec_y2))
                l2.setWidth(recurrent_width)
                l3 = Line(Point(rec_x2, rec_y2), Point(c_x2, rec_y2))
                l3.setWidth(recurrent_width)
                l3.setArrow('last')
                lines.extend([l1, l2, l3])
#                 # draw the arrowhead
#                 arrowSize = 5
#                 ax, ay = c_x2, rec_y2
#                 l1 = Line(Point(ax, ay), Point(ax+arrowSize, ay-arrowSize))
#                 l1.setWidth(recurrent_width)
#                 l2 = Line(Point(ax, ay), Point(ax+arrowSize, ay+arrowSize))
#                 l2.setWidth(recurrent_width)
#                 lines.extend([l1, l2])
        for l in lines:
            l.draw(self)
            l.canvas.tag_lower(l.id)
        # draw labels
        for label in labels:
            label.setFill(fgText)
            label.draw(self)
            # draw the text background
            bb_x1, bb_y1, bb_x2, bb_y2 = label.canvas.bbox(label.id)
            self.create_rectangle(bb_x1, bb_y1, bb_x2, bb_y2, fill=bgText, outline=bgText)
            label.canvas.tag_raise(label.id)


class VisualMatrix():
    def __init__(self, values=None, filename=None, title='', shape=0):
        # invert flag displays an inverted version of the image, but does not invert
        # the actual intensity values
        if filename is None and values is not None:
            assert type(values) is list and len(values) > 0, 'a list of values is required'
            for v in values:
                assert 0 <= v <= 1, 'image values must be in range 0-1'
            shape = self.validateShape(values, shape, None)
            self.rows, self.cols = shape
            self.normalized = values
            self.raw = [int(100*v) for v in values]
            self.maxval = 100
        elif filename is not None and values is None:
            assert shape is 0, 'shape is determined by PGM filename'
            f = open(filename)
            pgmType = f.readline().strip()
            if pgmType not in ['P2', 'P5']:
                raise IOError('file is not a valid PGM file')
            title = os.path.basename(filename)
            self.cols, self.rows = [int(v) for v in f.readline().split()]
            self.maxval = int(f.readline().strip())
            if pgmType == 'P5':
                self.raw = [ord(v) for v in f.read()]
            else:
                self.raw = [int(v) for v in f.read().split()]
            for v in self.raw:
                assert 0 <= v <= self.maxval, 'incorrect PGM file format'
            self.normalized = [float(v)/self.maxval for v in self.raw]
            f.close()
        else:
            raise AttributeError('must specify filename=<filename> or values=<vector>')
        self.rectangles = []
        self.title = title

    def render(self, canvas, scale=0, highlight=None, invert=False):
        self.scale = self.validateScale(scale, 10)
        self.width = self.cols * self.scale
        self.height = self.rows * self.scale
        self.invert = invert
        if highlight is not None:
            assert type(highlight) is int and self.rows == 1, \
                   'cannot highlight images with more than one row'
            assert 0 <= highlight < self.cols, 'highlight out of range'
        self.highlight = highlight
        self.rectangles = []
        for i in range(len(self.normalized)):
            x = (i % self.cols) * self.scale
            y = (i // self.cols) * self.scale
            grayLevel = int(255 * self.normalized[i])
            if self.invert:
                grayLevel = 255 - grayLevel
            fColor = oColor = 'gray%d' % grayLevel
            r = Rectangle((x, y), (x+self.scale, y+self.scale))
            if self.rows != 1 and self.cols != 1:
                r.noStroke()
            r.fill(Color(grayLevel, grayLevel, grayLevel))
            r.draw(canvas)
            self.rectangles.append(r)
        if self.highlight is not None:
            x = self.highlight * self.scale + 1
            r = Rectangle((x, 1), (x+self.scale-1, self.scale))
            r.fill(Color(255, 0, 0))
            r.draw(canvas)

    def __str__(self):
        s = '\ntitle:  %s\n' % self.title
        s += 'size:   %d rows, %d cols\n' % (self.rows, self.cols)
        s += 'maxval: %d\n' % self.maxval
        border = '+%s+\n' % ('-' * (2 * self.cols + 1))
        s += border
        palette = ' .,:+*O8@@'
        for r in range(self.rows):
            s += '| '
            for c in range(self.cols):
                i = r * self.cols + c
                if i >= len(self.normalized):
                    s += '  '
                else:
                    s += '%s ' % palette[int(self.normalized[i]*(len(palette)-1))]
            s = s + '|\n'
        s += border
        return s

    def setTitle(self, title):
        self.winfo_toplevel().title(title)
        self.title = title

    def updateImage(self, newValues, invert=False):
        # invert flag True displays an inverted version of the image,
        # but does not invert the intensity values themselves
        assert len(newValues) == len(self.rectangles), 'wrong number of values'
        for i in range(len(newValues)):
            assert 0 <= newValues[i] <= 1, 'image values must be in range 0-1'
            grayLevel = int(255 * newValues[i])
            if invert:
                grayLevel = 255 - grayLevel
            r = self.rectangles[i]
            r.fill(Color(grayLevel, grayLevel, grayLevel))
        # update graphics
        self.normalized = newValues
        self.raw = [int(100*v) for v in self.normalized]
        self.maxval = 100

    def invertImage(self):
        """"
        Sets the intensity values of the image to their inverse.
        """
        self.raw = [self.maxval-v for v in self.raw]
        self.normalized = [float(v)/self.maxval for v in self.raw]
        for i in range(len(self.normalized)):
            grayLevel = int(255 * self.normalized[i])
            r = self.rectangles[i]
            r.fill(Color(grayLevel, grayLevel, grayLevel))
        # update graphics

    def saveImage(self, pathname, pgmType='P5'):
        assert pgmType in ['P2', 'P5'], 'invalid PGM file type'
        f = open(pathname, mode='w')
        f.write('%s\n' % pgmType)
        f.write('%d %d\n' % (self.cols, self.rows))
        f.write('%d\n' % self.maxval)
        padding = []
        if len(self.raw) < self.rows * self.cols:
            padding = [0] * (self.rows * self.cols - len(self.raw))
        if pgmType == 'P5':
            f.write('%s' % ''.join([chr(v) for v in self.raw + padding]))
        else: # P2
            values = self.raw + padding
            i = 0
            for r in range(self.rows):
                for c in range(self.cols):
                    f.write('%d ' % values[i])
                    i += 1
                f.write('\n')
        f.close()

    def validateShape(self, values, shape, default):
        if shape is 0: shape = default
        if shape is None:
            shape = (1, len(values))
        elif type(shape) is int and 0 < shape <= len(values):
            cols = shape
            rows = int(math.ceil(float(len(values))/shape))
            shape = (rows, cols)
        elif type(shape) is tuple and len(shape) == 1:
            shape = (1, shape[0])
        assert type(shape) is tuple and len(shape) == 2, 'invalid shape: %s' % (shape,)
        (rows, cols) = shape
        assert rows * cols == len(values), \
               "can't display %d values with shape %s" % (len(values), shape)
        return shape

    def validateScale(self, scale, default):
        if scale is 0: scale = default
        if scale is None: scale = 10
        assert type(scale) is int and scale > 0, 'invalid scale: %s' % scale
        return scale

    def pretty(self, values, max=0, places=2):
        code = '%%.%df' % places
        if max > 0 and len(values) > max:
            return ' '.join([code % v for v in values[0:max]]) + ' ...'
        else:
            return ' '.join([code % v for v in values])
