using System;
using System.Collections.Generic;
using System.Windows.Media;


namespace CellUniverse.Models {

    internal sealed class ColorWorker {

        private const byte darkPoint = 0, lightPoint = 252;
        private const byte channelWidth = lightPoint - darkPoint;

        private Random random;

        internal ColorWorker() {
            random = new Random();
        }

        internal List<Color> GetTintsFromColor(Color color, int tintsCount) {

            int colorWidth = GetColorWidth(color);

            if (tintsCount <= 1 || tintsCount > channelWidth) {
                return new List<Color> { color };
            }

            List<Color> fullSequence = GenerateFullSequenceFromColor(color);
            List<Color> result = new List<Color>(tintsCount);

            var stepWidth = (double)channelWidth / tintsCount;

            for (int i = 0; i < tintsCount; i++) {
                int index = 0;
                try {
                    index = (int)Math.Round(stepWidth * i);
                }
                catch (Exception) {
                    index = (int)Math.Round(stepWidth * i) - 1;
                }
                result.Add(fullSequence[index]);
            }
            return result;
        }

        private List<Color> GenerateFullSequenceFromColor(Color color) {

            byte minInChannel  = GetMinValueOfTheChannel(color);
            List<Color> result = new List<Color>();

            for (int i = darkPoint; i < lightPoint; i++) {

                byte r = (byte)Truncate((color.R - minInChannel) + i, darkPoint, lightPoint);
                byte g = (byte)Truncate((color.G - minInChannel) + i, darkPoint, lightPoint);
                byte b = (byte)Truncate((color.B - minInChannel) + i, darkPoint, lightPoint);

                Color nextColor = Color.FromRgb(r, g, b);
                result.Add(nextColor);
            }
            return result;
        }

        private int Truncate(int value, int min, int max) {
            value = value >= min ? value : min;
            value = value <= max ? value : max;
            return value;
        }

        private int GetColorWidth(Color color) {
            return GetMaxValueOfTheChannel(color) - GetMinValueOfTheChannel(color);
        }

        private byte GetMinValueOfTheChannel(Color color) {
            return Math.Min(color.R, Math.Min(color.G, color.B));
        }

        private byte GetMaxValueOfTheChannel(Color color) {
            return Math.Max(color.R, Math.Max(color.G, color.B));
        }
    }
}

//Color.FromRgb(138, 198, 233)
//Color.FromRgb(0, 125, 168)
//Color.FromRgb(0, 49, 71)