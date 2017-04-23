using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace ASD.CellUniverse.Controls.Helpers {

    internal static class MeasureArrangeHelper {

        public static Size ComputeSize(Size availableSize, Size contentSize, Stretch stretch, StretchDirection stretchDirection) {
            Size scaleFactor = ComputeScaleFactor(availableSize, contentSize, stretch, stretchDirection);
            return new Size(contentSize.Width * scaleFactor.Width, contentSize.Height * scaleFactor.Height);
        }

        private static Size ComputeScaleFactor(Size availableSize, Size contentSize, Stretch stretch, StretchDirection stretchDirection) {

            var scaleX = 1.0;
            var scaleY = 1.0;

            var isConstrainedWidth = !double.IsPositiveInfinity(availableSize.Width);
            var isConstrainedHeight = !double.IsPositiveInfinity(availableSize.Height);

            if ((stretch == Stretch.Uniform || stretch == Stretch.UniformToFill || stretch == Stretch.Fill) && (isConstrainedWidth || isConstrainedHeight)) {

                scaleX = contentSize.Width == 0.0 ? 0.0 : availableSize.Width / contentSize.Width;
                scaleY = contentSize.Height == 0.0 ? 0.0 : availableSize.Height / contentSize.Height;

                if (!isConstrainedWidth) { scaleX = scaleY; }
                else if (!isConstrainedHeight) { scaleY = scaleX; }
                else {
                    switch (stretch) {

                        case Stretch.Uniform:
                            var minscale = scaleX < scaleY ? scaleX : scaleY;
                            scaleX = scaleY = minscale;
                            break;

                        case Stretch.UniformToFill:
                            var maxscale = scaleX > scaleY ? scaleX : scaleY;
                            scaleX = scaleY = maxscale;
                            break;

                        case Stretch.Fill: break;
                    }
                }
                switch (stretchDirection) {

                    case StretchDirection.UpOnly:
                        if (scaleX < 1.0) { scaleX = 1.0; }
                        if (scaleY < 1.0) { scaleY = 1.0; }
                        break;

                    case StretchDirection.DownOnly:
                        if (scaleX > 1.0) { scaleX = 1.0; }
                        if (scaleY > 1.0) { scaleY = 1.0; }
                        break;

                    case StretchDirection.Both: break;
                    default: break;
                }
            }
            return new Size(scaleX, scaleY);
        }
    }
}