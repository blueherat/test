"""Embed complete, editable font data in a native PowerPoint presentation.

EOT wrapper: https://www.w3.org/submissions/EOT/#Version3
PresentationML: https://learn.microsoft.com/en-us/dotnet/api/documentformat.openxml.presentation.embeddedfont
The original TTF bytes and embedding permission bits remain unchanged.
"""
import struct
from fontTools.ttLib import TTFont
from pptx.opc.package import Part
from pptx.opc.packuri import PackURI
from pptx.oxml.xmlchemy import OxmlElement
from pptx.oxml.ns import qn


def eot_data(path):
    data = path.read_bytes()
    with TTFont(path) as font:
        os2 = font['OS/2']
        # These supplied core fonts permit installable embedding.
        if os2.fsType != 0:
            raise ValueError('This figure embeds only fonts permitting installable embedding')
        panose = bytes(getattr(os2.panose, name) for name in (
            'bFamilyType', 'bSerifStyle', 'bWeight', 'bProportion', 'bContrast',
            'bStrokeVariation', 'bArmStyle', 'bLetterForm', 'bMidline', 'bXHeight'))
        header = struct.pack('<4I10sBBIHH11I', 0, len(data), 0x00020002, 0,
            panose, 1, os2.fsSelection & 1, os2.usWeightClass, os2.fsType, 0x504C,
            *(getattr(os2, 'ulUnicodeRange'+str(i)) for i in range(1, 5)),
            *(getattr(os2, 'ulCodePageRange'+str(i), 0) for i in range(1, 3)),
            font['head'].checkSumAdjustment, 0, 0, 0, 0)
        names = b''
        for name_id in (1, 2, 5, 4):
            name = font['name'].getDebugName(name_id).encode('utf-16le')
            names += struct.pack('<HH', 0, len(name)) + name
    # Empty document root/signature/EUDC fields; checksum of the empty root.
    suffix = struct.pack('<HHIIHHII', 0, 0, 0x50475342, 0, 0, 0, 0, 0)
    eot = bytearray(header + names + suffix + data)
    struct.pack_into('<I', eot, 0, len(eot))
    return bytes(eot)


def embed_family(prs, family, regular, bold):
    font_list = OxmlElement('p:embeddedFontLst')
    entry = OxmlElement('p:embeddedFont')
    font = OxmlElement('p:font')
    font.set('typeface', family)
    entry.append(font)
    for style, source in [('regular', regular), ('bold', bold)]:
        part = Part(PackURI('/ppt/fonts/'+style+'.fntdata'), 'application/x-fontdata',
                    prs.part.package, eot_data(source))
        relation = prs.part.relate_to(part,
            'http://schemas.openxmlformats.org/officeDocument/2006/relationships/font')
        item = OxmlElement('p:'+style)
        item.set(qn('r:id'), relation)
        entry.append(item)
    font_list.append(entry)
    root = prs._element
    root.insert(list(root).index(root.find(qn('p:defaultTextStyle'))), font_list)
    root.set('embedTrueTypeFonts', '1')
    root.set('saveSubsetFonts', '0')
