using System;
using System.Windows;
using System.Windows.Media.Imaging;

namespace ASD.CellUniverse.Controls.Helpers {

    internal sealed class WriteableContext : IDisposable {

        internal WriteableContext(WriteableBitmap bitmap) {
            this.bitmap = bitmap; bitmap.Lock();
        }

        public void Dispose() => bitmap.Unlock();

        internal unsafe void WriteRect(int posX, int posY, int width, int height, uint color) {

            var pBackBuffer = (long)bitmap.BackBuffer; // long address (8-Bytes) to x64 support

            for (var y = posY; y < posY + height; ++y) {
                for (var x = posX; x < posX + width; ++x) {

                    pBackBuffer += posY * bitmap.BackBufferStride;
                    pBackBuffer += posX * 4;

                    *((uint*)pBackBuffer) = color; // int / uint (4-byte) color value by address
                }
            }
            bitmap.AddDirtyRect(new Int32Rect(posX, posY, width, height));
        }
        private WriteableBitmap bitmap;
    }
}