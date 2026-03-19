import xml.etree.ElementTree as ET
import numpy as np

class KittiLabels:
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.objects = self._parse_xml()

    def _parse_xml(self):
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        
        objects = []
        for item in root.findall('tracklets/item'):
            obj = {
                'type': item.find('objectType').text,
                'h': float(item.find('h').text),
                'w': float(item.find('w').text),
                'l': float(item.find('l').text),
                'first_frame': int(item.find('first_frame').text),
                'poses': []
            }
            
            for pose in item.find('poses').findall('item'):
                obj['poses'].append({
                    'tx': float(pose.find('tx').text),
                    'ty': float(pose.find('ty').text),
                    'tz': float(pose.find('tz').text),
                    'ry': float(pose.find('ry').text)
                })
            objects.append(obj)
        return objects

    def get_boxes_for_frame(self, frame_idx):
        """Returns list of boxes active in this frame."""
        frame_boxes = []
        for obj in self.objects:
            relative_idx = frame_idx - obj['first_frame']
            if 0 <= relative_idx < len(obj['poses']):
                pose = obj['poses'][relative_idx]
                # Combine dimensions and pose
                box = {
                    'type': obj['type'],
                    'dims': [obj['h'], obj['w'], obj['l']],
                    'pos': [pose['tx'], pose['ty'], pose['tz']],
                    'yaw': pose['ry']
                }
                frame_boxes.append(box)
        return frame_boxes