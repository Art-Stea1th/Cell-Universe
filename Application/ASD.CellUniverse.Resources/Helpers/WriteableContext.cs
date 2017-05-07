using System;
using System.Windows;
using System.Windows.Media.Imaging;

namespace ASD.CellUniverse.Resources.Helpers {

    // Performance-Critical
    internal sealed class WriteableContext : IDisposable {

        private WriteableBitmap bitmap;
        private (int x, int y) size;

        private long addressOfBackBuffer;
        private long bytesWidth;
        private long sizeOfPixel;

        private unsafe uint this[int x, int y] {
            get => *((uint*)(addressOfBackBuffer + bytesWidth * y + sizeOfPixel * x));
            set => *((uint*)(addressOfBackBuffer + bytesWidth * y + sizeOfPixel * x)) = value;
        }

        internal WriteableContext(WriteableBitmap bitmap) {

            this.bitmap = bitmap;
            addressOfBackBuffer = bitmap.BackBuffer.ToInt64();

            size.x = bitmap.PixelWidth;
            size.y = bitmap.PixelHeight;

            bytesWidth = bitmap.BackBufferStride;
            sizeOfPixel = bytesWidth / size.x;

            bitmap.Lock();
        }

        public void Dispose() {
            bitmap.AddDirtyRect(new Int32Rect(0, 0, size.x, size.y));
            bitmap.Unlock();
        }

        internal void WriteCells(uint[,] cellCollection, int cellSize) {

            var countX = cellCollection.GetLength(0);
            var countY = cellCollection.GetLength(1);

            var opacity = cellSize < 5 ? 1.0f - (0.25f * (cellSize - 1)) : 0.0f;

            for (int cellX = 0, bitmapX = 0; cellX < countX; ++cellX, bitmapX += cellSize) {
                for (int cellY = 0, bitmapY = 0; cellY < countY; ++cellY, bitmapY += cellSize) {
                    WriteCellWithBorder(bitmapX, bitmapY, cellSize, cellCollection[cellX, cellY], opacity);
                }
            }
        }

        internal void WritePixels(uint[,] pixels) {
            for (var y = 0; y < pixels.GetLength(1); ++y) {
                for (var x = 0; x < pixels.GetLength(0); ++x) {
                    this[x, y] = pixels[x, y];
                }
            }
        }

        private void WriteCellWithBorder(int posX, int posY, int size, uint color, float borderOpacity) {

            var borderColor = color & 0x00FFFFFF | (uint)((color >> 24) * borderOpacity) << 24;

            for (var y = posY; y < posY + size; ++y) {
                for (var x = posX; x < posX + size; ++x) {
                    this[x, y] = x < posX + size - 1 && y < posY + size - 1
                        ? color
                        : borderColor;
                }
            }
        }
    }
}