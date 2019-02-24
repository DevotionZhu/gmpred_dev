import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import xml.etree.ElementTree
from os.path import join

from PIL import Image, ImageDraw


class GridMapCombiner():
    def __init__(self, base_out_path):
        self.base_out_path = base_out_path
        self.i = 1

    def do_combine(self, i1, i2, combined_name):
        images = [i1, i2]

        combined_image = self.append_images(images, direction='horizontal')
        combined_image_title = join(self.base_out_path, combined_name)
        combined_image.save(combined_image_title)
        self.i += 1


    def append_images(self, images, direction='horizontal',
                      bg_color=(255,255,255), aligment='center'):
        """
        Appends images in horizontal/vertical direction.

        Args:
            images: List of PIL images
            direction: direction of concatenation, 'horizontal' or 'vertical'
            bg_color: Background color (default: white)
            aligment: alignment mode if images need padding;
               'left', 'right', 'top', 'bottom', or 'center'

        Returns:
            Concatenated image as a new PIL image object.
        """
        widths, heights = zip(*(i.size for i in images))

        if direction=='horizontal':
            new_width = sum(widths)
            new_height = max(heights)
        else:
            new_width = max(widths)
            new_height = sum(heights)

        new_im = Image.new('RGB', (new_width, new_height), color=bg_color)


        offset = 0
        for im in images:
            if direction=='horizontal':
                y = 0
                if aligment == 'center':
                    y = int((new_height - im.size[1])/2)
                elif aligment == 'bottom':
                    y = new_height - im.size[1]
                new_im.paste(im, (offset, y))
                offset += im.size[0]
            else:
                x = 0
                if aligment == 'center':
                    x = int((new_width - im.size[0])/2)
                elif aligment == 'right':
                    x = new_width - im.size[0]
                new_im.paste(im, (x, offset))
                offset += im.size[1]

        return new_im


    def do_point_transformation(self, x, y):
        y = y * -1
        x = abs(int(x) - self.max_x)
        y = int(y) - abs(self.min_y)

        return x, y


    def get_points_from_lane(self, lane):
        points = []
        for point in lane.findall('point'):
            x = float(point.find('x').text)
            y = float(point.find('y').text)

            x, y = self.do_point_transformation(x, y)

            points.append((x, y))

        return points


    def get_points(self):
        lane_points = []
        e = xml.etree.ElementTree.parse('lanelets.txt').getroot()
        for lanelet in e.findall('lanelet'):
            leftbound = lanelet.find('leftBound')
            rightbound = lanelet.find('rightBound')
            lane_points.append(self.get_points_from_lane(leftbound))
            lane_points.append(self.get_points_from_lane(rightbound))
        return lane_points


    def draw_line(self, image, x1, y1, x2, y2):
        draw = ImageDraw.Draw(image)
        draw.line((x2, y2, x1, y1), fill=150)
        # draw.line((y2, x2, y1, x1), fill=150)

        image.save('temp.jpeg')

        return image


    def draw_lanes(self, image):
        lane_points = self.get_points()
        for lane in lane_points:
            for i in range(0, len(lane) - 1):
                point = lane[i]
                x1 = point[0]
                y1 = point[1]

                point_1 = lane[i+1]

                x2 = point_1[0]
                y2 = point_1[1]

                image = self.draw_line(image, x1, y1, x2, y2)
                # image.show()

        return image


    def draw_background(self, image):
        background_img = Image.open('icons/google_earth_mp10_2.png')

        print (image.size)
        print (background_img.size)
        return image


    def draw_cars(self, image):
        lane_points = self.get_points()
        car_img = Image.open('icons/car.png')
        for lane in lane_points:
            for i in range(0, len(lane) - 1):
                point = lane[i]
                x1 = point[0]
                y1 = point[1]
                offset = (int(y1), int(x1))
                image.paste(car_img, offset)

        return image


    def draw_radar(self, image):
        x, y = self.do_point_transformation(0, 0)
        const = 5

        image = self.draw_line(image, x, y - const, x - const, y + const)
        image = self.draw_line(image, x - const, y - const, x, y + const)

        return image






