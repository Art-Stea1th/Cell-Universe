using System;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace ASD.CellUniverse.Converters {

    [ValueConversion(typeof(bool[,]), typeof(byte[]))]
    public class TempBool2dToByteArrayConverter : IValueConverter {

        public object Convert(object value, Type targetType, object parameter, CultureInfo culture) {

            var array = value as bool[,];

            var trueColor = Colors.DodgerBlue;
            var falseColor = Colors.Transparent;

            var width = array.GetLength(0);
            var height = array.GetLength(1);

            var colors = new Color[width, height];

            for (var y = 0; y < height; y++) {
                for (var x = 0; x < width; x++) {
                    colors[x, y] = array[x, y] ? trueColor : falseColor;
                }
            }

            var bytes = new byte[width * height * sizeof(int) / sizeof(byte)];

            for (var y = 0; y < height; ++y) {
                for (var x = 0; x < width; ++x) {
                    SetColorToByteArray(bytes, GetLinearIndex(x, y, width) * sizeof(int), colors[x, y]);
                }
            }
            return bytes;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
            => throw new NotSupportedException();

        private int GetLinearIndex(int x, int y, int width) {
            return width * y + x;
        }

        private void SetColorToByteArray(byte[] byteArray, int startIndex, Color color) {
            byteArray[startIndex] = color.B;
            byteArray[++startIndex] = color.G;
            byteArray[++startIndex] = color.R;
            byteArray[++startIndex] = color.A;
        }
    }
}
