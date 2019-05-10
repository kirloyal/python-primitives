import sys
import os
import xml.etree.ElementTree as ET

root = ET.Element("root")
sub1 = ET.SubElement(root, "sub", att='asdf')
ET.SubElement(sub1, "txt", c='a123').text = "hello"
ET.SubElement(sub1, "txt", d='432').text = "world"
sub2 = ET.SubElement(root, "sub", att='sdf')
blar = ET.SubElement(root, "blar", att='asdf')
ET.SubElement(blar, "txt1", gg='a123').text = "xml1"
ET.SubElement(blar, "txt2", d='432').text = "xml2"


tree = ET.ElementTree(root)
tree.write('tmp/tmp.xml', encoding="utf-8", xml_declaration=True)

#%%

tree = ET.parse('tmp/tmp.xml')
root = tree.getroot()

for track in root.findall('sub'):
    print(track.attrib['att']) # == print(track.get('att'))
    

for track in root.find('sub').findall('txt'):
    print(track.text, track.attrib)

