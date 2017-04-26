using System.Windows;

namespace ASD.CellUniverse.Controls.Helpers {

    internal static class MeasureArrangeHelper {

        public static Size ComputeSize(Size availableSize, Size contentSize) {
            Size scaleFactor = ComputeScaleFactor(availableSize, contentSize);
            return new Size(contentSize.Width * scaleFactor.Width, contentSize.Height * scaleFactor.Height);
        }

        private static Size ComputeScaleFactor(Size availableSize, Size contentSize) {

            double scaleX = 1.0, scaleY = 1.0;

            var isConstrainedWidth = !double.IsPositiveInfinity(availableSize.Width);
            var isConstrainedHeight = !double.IsPositiveInfinity(availableSize.Height);

            if (isConstrainedWidth || isConstrainedHeight) {

                scaleX = contentSize.Width == 0.0 ? 0.0 : availableSize.Width / contentSize.Width;
                scaleY = contentSize.Height == 0.0 ? 0.0 : availableSize.Height / contentSize.Height;

                if (!isConstrainedWidth) {
                    scaleX = scaleY;
                }
                else if (!isConstrainedHeight) {
                    scaleY = scaleX;
                }
                else {
                    var minscale = scaleX < scaleY ? scaleX : scaleY;
                    scaleX = scaleY = minscale;
                }
            }
            return new Size(scaleX, scaleY);
        }
    }
}