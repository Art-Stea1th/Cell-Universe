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

            var pBackBuffer = (long)bitmap.BackBuffer;

            for (var y = posY; y < posY + height; ++y) {
                for (var x = posX; x < posX + width; ++x) {

                    pBackBuffer += posY * bitmap.BackBufferStride;
                    pBackBuffer += posX * 4;

                    *((long*)pBackBuffer) = color;
                }
            }
            bitmap.AddDirtyRect(new Int32Rect(posX, posY, width, height));
        }
        private WriteableBitmap bitmap;
    }
}