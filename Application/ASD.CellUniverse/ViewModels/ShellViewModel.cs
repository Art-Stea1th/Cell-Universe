using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ASD.CellUniverse.ViewModels {

    using Infrastructure.Controllers;
    using Infrastructure.Interfaces;
    using Infrastructure.MVVM;

    public sealed class ShellViewModel : BindableBase {

        private IMainController controller;

        private WriteableBitmap pixelData;

        public ImageSource PixelData => pixelData;

        public ICommand Play => controller.Play;
        public ICommand Pause => controller.Pause;
        public ICommand Stop => controller.Stop;

        public ShellViewModel() {

            controller = new ApplicationStateMachine();


            var field = CreateField(161, 90);
            pixelData = NewBitmapFrom(field);
        }

        private Color[,] CreateField(int width, int height) {

            var black = true;
            var colors = new Color[width, height];

            for (var y = 0; y < height; y++) {
                for (var x = 0; x < width; x++) {
                    if (black) { colors[x, y] = Colors.Black; }
                    else { colors[x, y] = Colors.DodgerBlue; }
                    black = !black;
                }
            }
            return colors;
        }

        private WriteableBitmap NewBitmapFrom(Color[,] colors) {

            int width = colors.GetLength(0), height = colors.GetLength(1);
            var bytes = new byte[width * height * sizeof(int) / sizeof(byte)];

            for (var y = 0; y < height; ++y) {
                for (var x = 0; x < width; ++x) {
                    SetColorToByteArray(bytes, GetLinearIndex(x, y, width) * sizeof(int), colors[x, y]);
                }
            }

            var result = new WriteableBitmap(width, height, 96.0, 96.0, PixelFormats.Bgra32, null);
            result.WritePixels(new Int32Rect(0, 0, width, height), bytes, result.PixelWidth * sizeof(int), 0);

            return result;
        }

        private int GetLinearIndex(int x, int y, int width) {
            return width * y + x;
        }

        private void SetColorToByteArray(byte[] viewportByteArray, int startIndex, Color color) {
            viewportByteArray[startIndex] = color.B;
            viewportByteArray[++startIndex] = color.G;
            viewportByteArray[++startIndex] = color.R;
            viewportByteArray[++startIndex] = color.A;
        }
    }
}