﻿using System;
using System.Windows;
using System.Windows.Media.Imaging;

namespace ASD.CellUniverse.Resources.Helpers {

    // Performance-Critical, Class is ultra optimized
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

        internal unsafe void WriteRect(int posX, int posY, int width, int height, uint color) {
            for (var y = posY; y < posY + height; ++y) {
                for (var x = posX; x < posX + width; ++x) {
                    this[x, y] = color;
                }
            }
        }

        internal unsafe void WriteGrid(int cellSize, int lineThickness, uint color) {
            var step = cellSize + lineThickness;
            for (var y = 0; y < size.y; ++y) {
                for (var x = 0; x < size.x; ++x) {
                    if (x % step == 0 || y % step == 0) {
                        this[x, y] = color;
                    }
                }
            }
        }
    }
}