using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ASD.CellUniverse.Resources.Helpers {

    using Extensions;

    internal sealed class WriteableContext : IDisposable {

        internal WriteableContext(WriteableBitmap bitmap) {
            this.bitmap = bitmap;
            bitmap.Lock();
        }

        public void Dispose() {
            if (rangeChanged) {
                bitmap.AddDirtyRect(new Int32Rect(X.min, Y.min, X.max - X.min + 1, Y.max - Y.min + 1));
            }
            bitmap.Unlock();
        }

        internal unsafe Color this[int x, int y] {
            get => ValueBy(PointerBy(AddressBy(x, y))).Argb32();
            set {
                if (x < 0 || x >= bitmap.PixelWidth) {
                    throw new IndexOutOfRangeException($"{x}");
                }
                if (y < 0 || y >= bitmap.PixelHeight) {
                    throw new IndexOutOfRangeException($"{y}");
                }
                RefBy(PointerBy(AddressBy(x, y))) = value.Bgra32();
                UpdateRange(x, y);
            }
        }

        internal unsafe void WritePixelSequence(IEnumerable<(int x, int y)> sequence, Color color) {
            foreach (var pixel in sequence) {
                this[pixel.x, pixel.y] = color;
            }
        }

        internal unsafe void WritePixelSequence(IEnumerable<(int x, int y, Color color)> sequence) {
            foreach (var pixel in sequence) {
                this[pixel.x, pixel.y] = pixel.color;
            }
        }

        internal unsafe void WriteRect(int posX, int posY, int width, int height, Color color) {
            for (var y = posY; y < posY + height; ++y) {
                for (var x = posX; x < posX + width; ++x) {
                    this[x, y] = color;
                }
            }
        }

        internal unsafe void WriteRectSequence(IEnumerable<(int x, int y)> sequence, int rectSize, Color color) {
            foreach (var position in sequence) {
                WriteRect(position.x, position.y, rectSize, rectSize, color);
            }
        }

        internal unsafe void WriteRectSequence(IEnumerable<(int x, int y, Color color)> sequence, int rectSize) {
            foreach (var position in sequence) {
                WriteRect(position.x, position.y, rectSize, rectSize, position.color);
            }
        }

        internal unsafe void WriteGrid(int cellSize, int lineThickness, Color color) {
            for (var y = 0; y < bitmap.PixelHeight; ++y) {
                for (var x = 0; x < bitmap.PixelWidth; ++x) {
                    if (x % (cellSize + lineThickness) == 0 || y % (cellSize + lineThickness) == 0) {
                        this[x, y] = color;
                    }
                }
            }
        }

        private void UpdateRange(int nextX, int nextY) {
            if (nextX > X.max) { X.max = nextX; }
            else if (nextX < X.min) { X.min = nextX; }

            if (nextY > Y.max) { Y.max = nextY; }
            else if (nextY < Y.min) { Y.min = nextY; }

            rangeChanged = true;
        }

        // --- int / uint (4-byte) color value by address
        private unsafe uint ValueBy(uint* pointer) => *pointer;
        private unsafe ref uint RefBy(uint* pointer) => ref *pointer;
        private unsafe uint* PointerBy(long address) => (uint*)address;

        // --- long address (8-Bytes) to x64 support
        private long AddressBy(int x, int y) => bitmap.BackBuffer.ToInt64() + Offset(x, y);
        //private long Offset(int x, int y) => (long)bitmap.BackBufferStride * y + (long)bitmap.PixelWidth * x;
        private long Offset(int x, int y) => (long)bitmap.BackBufferStride * y + (long)PixelTypeSize * x;

        // --- to support different PixelFormats
        private int PixelTypeSize => bitmap.BackBufferStride / bitmap.PixelWidth;


        // ---
        private bool rangeChanged = false;

        private (int min, int max) X = (min: int.MaxValue, max: int.MinValue);
        private (int min, int max) Y = (min: int.MaxValue, max: int.MinValue);

        private WriteableBitmap bitmap;
    }
}