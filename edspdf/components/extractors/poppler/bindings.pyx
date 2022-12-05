from pathlib import Path

from cpython cimport bool as PyBool
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.string cimport string

import os
import sys

import six

ctypedef bool GBool
ctypedef int Goffset


cdef extern from * namespace "polyfill":
    """
    namespace polyfill {

    template <typename T>
    inline typename std::remove_reference<T>::type&& move(T& t) {
        return std::move(t);
    }

    template <typename T>
    inline typename std::remove_reference<T>::type&& move(T&& t) {
        return std::move(t);
    }

    }  // namespace polyfill
    """
    cdef T move[T](T)


#cdef extern from "cpp/poppler-version.h" namespace "poppler":
#    cdef string version_string()
#
#def poppler_version():
#    return version_string()


cdef extern from "poppler/GlobalParams.h":
    unique_ptr[GlobalParams] globalParams
    cdef cppclass GlobalParams:
        pass


globalParams.reset(new GlobalParams())


cdef extern from "goo/GooString.h":
    cdef cppclass GooString:
        GooString()
        GooString(const char *sA)
        int getLength()
        string toStr()
        char getChar(int i)


cdef extern from "poppler/OutputDev.h":
    cdef cppclass OutputDev:
        pass


cdef extern from 'poppler/Annot.h':
    cdef cppclass Annot:
        pass


cdef extern from 'poppler/Dict.h':
    cdef cppclass Dict:
        int getLength()
        char *getKey(int i)
        Object getVal(int i)


cdef extern from 'poppler/Object.h':
    cdef cppclass Object:
        Object()
        GBool isDict()
        Dict *getDict()
        GBool isString()
        unique_ptr[GooString] getString()


cdef extern from "poppler/PDFDoc.h":
    cdef cppclass PDFDoc:
        PDFDoc()
        GBool isOk()
        int getNumPages()
        void displayPage(
              OutputDev *out, int page,
              double hDPI, double vDPI, int rotate,
              GBool useMediaBox, GBool crop, GBool printing,
              GBool (*abortCheckCbk)(void *data)=NULL,
              void *abortCheckCbkData=NULL,
              GBool (*annotDisplayDecideCbk)(Annot *annot, void *user_data)=NULL,
              void *annotDisplayDecideCbkData=NULL, GBool copyXRef=False
        )
        void displayPages(
              OutputDev *out, int firstPage, int lastPage,
              double hDPI, double vDPI, int rotate,
              GBool useMediaBox, GBool crop, GBool printing,
              GBool (*abortCheckCbk)(void *data)=NULL,
              void *abortCheckCbkData=NULL,
              GBool (*annotDisplayDecideCbk)(Annot *annot, void *user_data)=NULL,
              void *annotDisplayDecideCbkData=NULL
        )
        double getPageMediaWidth(int page)
        double getPageMediaHeight(int page)
        unique_ptr[GooString] readMetadata()
        Object getDocInfo()


cdef extern from "poppler/PDFDocFactory.h":
    cdef cppclass PDFDocFactory:
        PDFDocFactory()
        unique_ptr[PDFDoc] createPDFDoc(const GooString & uri)  #, GooString* ownerPassword, GooString* userPassword, void *guiDataA)


cdef extern from "poppler/TextOutputDev.h":
    cdef cppclass TextOutputDev:
        TextOutputDev(char *fileName, GBool physLayoutA, double fixedPitchA,
                      GBool rawOrderA, GBool append)
        TextPage *takeText()

    cdef cppclass TextPage:
        void incRefCnt()
        void decRefCnt()
        TextFlow *getFlows()
        unique_ptr[GooString] getText(double xMin, double yMin, double xMax, double yMax) const

    cdef cppclass TextFlow:
        TextFlow *getNext()
        TextBlock *getBlocks()

    cdef cppclass TextBlock:
        TextBlock *getNext()
        TextLine *getLines()

    cdef cppclass TextLine:
        TextWord *getWords()
        TextLine *getNext()

    cdef cppclass TextWord:
        TextWord *getNext()
        unique_ptr[GooString] getText()
        void getCharBBox(int charIdx, double *xMinA, double *yMinA, double *xMaxA, double *yMaxA)
        void getBBox(double *xMinA, double *yMinA, double *xMaxA, double *yMaxA)
        GBool hasSpaceAfter()
        const TextFontInfo *getFontInfo(int idx)

    cdef cppclass TextFontInfo:
        const unique_ptr[GooString] getFontName()
        GBool isFixedWidth()
        GBool isSerif()
        GBool isSymbolic()
        GBool isItalic()
        GBool isBold()

cdef extern from "poppler/Stream.h":
    cdef cppclass MemStream:
        MemStream(const char *bufA, Goffset startA, Goffset lengthA, Object & & dictA)
        GBool isFixedWidth()
        GBool isSerif()
        GBool isSymbolic()
        GBool isItalic()
        GBool isBold()


# cdef extern from "utils/ImageOutputDev.h":
#     cdef cppclass ImageOutputDev:
#         ImageOutputDev(char *fileRootA, GBool pageNamesA, GBool dumpJPEGA)
#         void enablePNG(GBool png)
#         void enableTiff(GBool tiff)
#         void enableJpeg(GBool jpeg)
#         void enableJpeg2000(GBool jp2)
#         void enableJBig2(GBool jbig2)
#         void enableCCITT(GBool ccitt)


cdef double RESOLUTION = 72

cdef void test():
    cdef unique_ptr[PDFDoc] doc = PDFDocFactory().createPDFDoc(
        GooString("/Users/perceval/Downloads/letter.pdf")  #, NULL, NULL, NULL
    )
    cdef TextOutputDev *dev = new TextOutputDev(NULL, False, 0, False, False)
    deref(doc).displayPage(
        <OutputDev *> dev, 1, RESOLUTION, RESOLUTION, 0, True, False, False
    )
    del dev

cdef class Document:
    cdef:
        unique_ptr[PDFDoc] _doc
        # ImageOutputDev *imgOut
        int _pg
        PyBool phys_layout
        double fixed_pitch
        bool extract_style

    def __cinit__(self, object filename, PyBool phys_layout=False,
                  double fixed_pitch=0, bool extract_style=False):
        cdef:
            TextOutputDev *dev

        self._pg = 0
        self.phys_layout = phys_layout
        self.fixed_pitch = fixed_pitch
        self.extract_style = extract_style
        # Sanity checks
        if isinstance(filename, Path):
            filename = str(filename)
            if not isinstance(filename, (six.binary_type, six.text_type)):
                raise ValueError("Invalid path: " + repr(filename))
            if isinstance(filename, six.binary_type):
                filename = filename.decode(sys.getfilesystemencoding())
            if not os.path.exists(filename) or not os.path.isfile(filename):
                raise IOError("Not a valid file path: " + filename)
            self._doc = move(PDFDocFactory().createPDFDoc(
                GooString(filename.encode(sys.getfilesystemencoding()))  #, NULL, NULL, NULL
            ))

            if not deref(self._doc).isOk():
                raise IOError("Error opening file: " + filename)
        elif isinstance(filename, bytes):
            self._doc = move(self.from_bytes(filename))
            if not deref(self._doc).isOk():
                raise IOError("Error reading bytes")
        else:
            raise Exception("Could not understand the format of input:", type(filename))

        #test()

    # def __dealloc__(self):
    #     if self._doc != NULL:
    #         del self._doc

    cdef unique_ptr[PDFDoc] from_bytes(self, bytes content):

        cdef:
            MemStream *stream
            char * c_chars
            unique_ptr[PDFDoc] doc

        c_chars = content
        stream = new MemStream(c_chars, 0, len(content), Object())

        doc = make_unique[PDFDoc](stream)

        return move(doc)

    property no_of_pages:
        def __get__(self):
            return deref(self._doc).getNumPages()

    cdef void render_page(self, int page_no, OutputDev *dev):
        deref(self._doc).displayPage(
            dev, page_no, RESOLUTION, RESOLUTION, 0, True, False, False
        )

    cdef object get_page_size(self, page_no):
        cdef double w, h
        w = deref(self._doc).getPageMediaWidth(page_no)
        h = deref(self._doc).getPageMediaHeight(page_no)
        return w, h

    def __iter__(self):
        return self

    def get_page(self, int pg):
        return Page(pg, self)

    def __next__(self):
        if self._pg >= self.no_of_pages:
            self._pg = 0
            raise StopIteration()
        self._pg += 1
        return self.get_page(self._pg)

cdef class Page:
    cdef:
        int page_no
        TextPage *page
        Document doc
        const TextFlow *curr_flow

    def __cinit__(self, int page_no, Document doc):
        cdef TextOutputDev *dev
        self.page_no = page_no
        self.doc = doc
        dev = new TextOutputDev(NULL, doc.phys_layout, doc.fixed_pitch, False, False)
        doc.render_page(page_no, <OutputDev *> dev)
        self.page = dev.takeText()
        del dev
        self.curr_flow = self.page.getFlows()

    def __dealloc__(self):
        if self.page != NULL:
            self.page.decRefCnt()

    def __iter__(self):
        return self

    def __next__(self):
        cdef Flow flow
        if not self.curr_flow:
            raise StopIteration()
        flow = Flow(self, self.doc.extract_style)
        self.curr_flow = self.curr_flow.getNext()
        return flow

    property page_no:
        def __get__(self):
            return self.page_no

    property size:
        """Size of page as (width, height)"""
        def __get__(self):
            return self.doc.get_page_size(self.page_no)

    # property lines:
    #     def __get__(self):
    #         lines = []
    #         pageText = deref(self.page.getText(sys.float_info.min, sys.float_info.min, sys.float_info.max, sys.float_info.max)).toStr()
    #         for line in pageText.decode('UTF-8')splitlines():
    #             lines.append(line.strip())
    #         return lines

    # def extract_images(self, path, prefix):
    #     self.doc.extract_images(
    #         path=path, prefix=prefix,
    #         first_page=self.page_no, last_page=self.page_no
    #     )

cdef class Flow:
    cdef:
        const TextFlow *flow
        const TextBlock *curr_block
        bool extract_style

    def __cinit__(self, Page pg, bool extract_style):
        self.flow = pg.curr_flow
        self.extract_style = extract_style
        self.curr_block = self.flow.getBlocks()

    def __iter__(self):
        return self

    def __next__(self):
        cdef Block b
        if not self.curr_block:
            raise StopIteration()
        b = Block(self, self.extract_style)
        self.curr_block = self.curr_block.getNext()
        return b

cdef class Block:
    cdef:
        const TextBlock *block
        const TextLine *curr_line
        bool extract_style

    def __cinit__(self, Flow flow, bool extract_style):
        self.block = flow.curr_block
        self.curr_line = self.block.getLines()
        self.extract_style = extract_style

    def __iter__(self):
        return self

    def __next__(self):
        cdef Line line
        if not self.curr_line:
            raise StopIteration()
        line = Line(self, self.extract_style)
        self.curr_line = self.curr_line.getNext()
        return line

cdef class Line:
    cdef:
        list _styles
        const TextLine *line
        unicode _text
        double _x0
        double _y0
        double _x1
        double _y1

    def __cinit__(self, Block block, bool extract_style):
        self.line = block.curr_line

    def __init__(self, Block block, bool extract_style):
        self._styles = []
        self._text = u''
        if extract_style:
            self._get_text_with_style()
        else:
            self._get_text()

    def _get_text(self):
        cdef:
            const TextWord *word
            const TextWord *last_word
            list words = []
            double dummy

        word = self.line.getWords()
        last_word = word
        word.getBBox(
            &self._x0, &self._y0,
            &dummy, &dummy,
        )
        while word:
            string = deref(word.getText()).toStr().decode('UTF-8')
            words.append(string)

            del string
            # add space after word if necessary
            if word.hasSpaceAfter():
                words.append(u' ')
            last_word = word
            word = word.getNext()

        last_word.getBBox(
            &dummy, &dummy,
            &self._x1, &self._y1,
        )
        self._text = u''.join(words)
        self._styles = None

    def _get_text_with_style(self):
        cdef:
            const TextWord *word
            const TextWord *last_word
            #GooString *string
            str font_name
            const TextFontInfo *font_info
            list words = []
            Style new_style
            Style last_style
            double dummy

        word = self.line.getWords()
        last_word = word
        offset = 0
        last_offset = 0
        new_style = Style(
            begin=0,
            end=0,
            is_bold=False,
            is_italic=False,
            font_name="")
        last_style = None
        styles = []
        while word:
            string = deref(word.getText()).toStr().decode('UTF-8')
            words.append(string)

            if offset == 0:
                word.getBBox(
                    &self._x0, &self._y0,
                    &dummy, &dummy
                )

            for char_idx in range(len(string)):
                font_info = word.getFontInfo(char_idx)
                font_name = (
                    deref(font_info.getFontName())
                    .toStr()
                    .decode('UTF-8')
                )
                lower_font_name = font_name.lower()

                new_style._begin = offset
                new_style._end = offset + 1
                new_style._is_bold = font_info.isBold() or ("bold" in lower_font_name)
                new_style._is_italic = font_info.isItalic() or ("italic" in lower_font_name)
                new_style._font_name = font_name
                offset += 1

                # (
                #     font_name
                #     .replace("bold", "")
                #     .replace("italic", "")
                #     .replace("serif", "")
                #     .replace("monospace", "")
                # )

                if last_style is not None and last_style.can_merge(new_style):
                    # extend the char boundaries
                    last_style._end = new_style._end
                else:
                    if last_style is not None:
                        styles.append(last_style)
                    # move the new style
                    last_style = Style(
                        begin=new_style._begin,
                        end=new_style._end,
                        is_bold=new_style._is_bold,
                        is_italic=new_style._is_italic,
                        font_name=new_style._font_name,
                    )
            del string
            # add space after word if necessary
            if word.hasSpaceAfter():
                words.append(u' ')
                offset += 1
            last_word = word
            word = word.getNext()

        last_word.getBBox(
            &dummy, &dummy,
            &self._x1, &self._y1
        )
        styles.append(last_style)
        self._text = u''.join(words)
        self._styles = styles

    property text:
        def __get__(self):
            return self._text

    property styles:
        def __get__(self):
            return self._styles

    property x0:
        def __get__(self):
            return self._x0

    property y0:
        def __get__(self):
            return self._y0

    property x1:
        def __get__(self):
            return self._x1

    property y1:
        def __get__(self):
            return self._y1

cdef class Style:
    cdef:
        int _begin
        int _end
        PyBool _is_bold
        PyBool _is_italic
        str _font_name
        # PyBool _is_serif
        # PyBool _is_monospace

    def __cinit__(
          self,
          int begin,
          int end,
          PyBool is_bold,
          PyBool is_italic,
          str font_name,
    ):
        self._begin = begin
        self._end = end
        self._is_bold = is_bold
        self._is_italic = is_italic
        self._font_name = font_name

    cdef bool can_merge(self, Style other):
        return (
              (self._is_bold == other._is_bold) and
              (self._is_italic == other._is_italic) and
              # (self._is_serif == other._is_serif) and
              # (self._is_monospace == other._is_monospace) and
              (self._font_name == other._font_name)
        )

    property is_bold:
        def __get__(self):
            return self._is_bold

    property is_italic:
        def __get__(self):
            return self._is_italic

    #    property is_serif:
    #        def __get__(self):
    #            return self._is_serif
    #
    #    property is_monospace:
    #        def __get__(self):
    #            return self._is_monospace

    property font_name:
        def __get__(self):
            return self._font_name

    property begin:
        def __get__(self):
            return self._begin

    property end:
        def __get__(self):
            return self._end
