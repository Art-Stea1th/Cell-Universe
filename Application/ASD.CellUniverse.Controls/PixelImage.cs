using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using ASD.CellUniverse.Controls.Helpers;

namespace ASD.CellUniverse.Controls {

    public class PixelImage : FrameworkElement {

        public ImageSource Source {
            get => (ImageSource)GetValue(SourceProperty);
            set => SetValue(SourceProperty, value);
        }

        public Stretch Stretch {
            get => (Stretch)GetValue(StretchProperty);
            set => SetValue(StretchProperty, value);
        }

        public StretchDirection StretchDirection {
            get => (StretchDirection)GetValue(StretchDirectionProperty);
            set => SetValue(StretchDirectionProperty, value);
        }

        public static readonly DependencyProperty SourceProperty = DependencyProperty.Register(
            "Source", typeof(ImageSource), typeof(PixelImage), new FrameworkPropertyMetadata(
                null, FrameworkPropertyMetadataOptions.AffectsMeasure | FrameworkPropertyMetadataOptions.AffectsRender));

        public static readonly DependencyProperty StretchProperty =
            Viewbox.StretchProperty.AddOwner(typeof(PixelImage));

        public static readonly DependencyProperty StretchDirectionProperty =
            Viewbox.StretchDirectionProperty.AddOwner(typeof(PixelImage));

        static PixelImage() {

            Style style = CreateDefaultStyles();
            StyleProperty.OverrideMetadata(typeof(PixelImage), new FrameworkPropertyMetadata(style));

            StretchProperty.OverrideMetadata(
                typeof(PixelImage), new FrameworkPropertyMetadata(
                    Stretch.Uniform, FrameworkPropertyMetadataOptions.AffectsMeasure));

            StretchDirectionProperty.OverrideMetadata(
                typeof(PixelImage), new FrameworkPropertyMetadata(
                    StretchDirection.Both, FrameworkPropertyMetadataOptions.AffectsMeasure));
        }

        private static void OnSourceChanged(DependencyObject d, DependencyPropertyChangedEventArgs e) {
            //var image = (PixelImage)d;
            //var oldValue = (ImageSource)e.OldValue;
            //var newValue = (ImageSource)e.NewValue;
        }

        private static Style CreateDefaultStyles() {
            var style = new Style(typeof(PixelImage), null);
            style.Setters.Add(new Setter(FlowDirectionProperty, FlowDirection.LeftToRight));
            style.Seal();
            return style;
        }

        protected override void OnRender(DrawingContext dc) {
            RenderOptions.SetBitmapScalingMode(this, BitmapScalingMode.NearestNeighbor);
            dc.DrawImage(Source, new Rect(new Point(), RenderSize));
        }

        protected override Size MeasureOverride(Size constraint) {
            return ComputeSize(constraint);
        }

        protected override Size ArrangeOverride(Size arrangeSize) {
            return ComputeSize(arrangeSize);
        }

        private Size ComputeSize(Size availableSize) {
            ImageSource imageSource = Source;
            if (imageSource == null) {
                return new Size();
            }
            var contentSize = new Size(imageSource.Width, imageSource.Height);
            return MeasureArrangeHelper.ComputeSize(availableSize, contentSize, Stretch, StretchDirection);
        }        
    }
}