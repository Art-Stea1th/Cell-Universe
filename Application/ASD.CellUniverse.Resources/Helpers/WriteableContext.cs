using System;
using System.Windows;
using System.Windows.Media.Imaging;

namespace ASD.CellUniverse.Resources.Helpers {

    // Performance-Critical, Class is optimized
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

        internal unsafe void WriteCells(uint[,] cellCollection, int cellSize) {

            var countX = cellCollection.GetLength(0);
            var countY = cellCollection.GetLength(1);

            for (int cellX = 0, bitmapX = 0; cellX < countX; ++cellX, bitmapX += cellSize) {
                for (int cellY = 0, bitmapY = 0; cellY < countY; ++cellY, bitmapY += cellSize) {
                    WriteRect(bitmapX, bitmapY, cellSize, cellSize, cellCollection[cellX, cellY]);
                }
            }
        }

        internal unsafe void WriteRect(int posX, int posY, int width, int height, uint color) {
            for (var y = posY; y < posY + height; ++y) {
                for (var x = posX; x < posX + width; ++x) {
                    this[x, y] = color;
                }
            }
        }
    }
}