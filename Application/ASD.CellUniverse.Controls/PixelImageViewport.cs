using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using ASD.CellUniverse.Controls.Helpers;

namespace ASD.CellUniverse.Controls {

    public class PixelImageViewport : FrameworkElement {

        public IEnumerable<Point> Source {
            get => (IEnumerable<Point>)GetValue(SourceProperty);
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

        public Brush Background {
            get => (Brush)GetValue(BackgroundProperty);
            set => SetValue(BackgroundProperty, value);
        }

        public Brush Foreground {
            get => (Brush)GetValue(ForegroundProperty);
            set => SetValue(ForegroundProperty, value);
        }

        public static readonly DependencyProperty SourceProperty = DependencyProperty.Register(
            "Source", typeof(IEnumerable<Point>), typeof(PixelImageViewport), new FrameworkPropertyMetadata(
                null, FrameworkPropertyMetadataOptions.AffectsMeasure | FrameworkPropertyMetadataOptions.AffectsRender));

        public static readonly DependencyProperty StretchProperty =
            Viewbox.StretchProperty.AddOwner(typeof(PixelImageViewport));

        public static readonly DependencyProperty StretchDirectionProperty =
            Viewbox.StretchDirectionProperty.AddOwner(typeof(PixelImageViewport));

        public static readonly DependencyProperty BackgroundProperty =
            Panel.BackgroundProperty.AddOwner(typeof(PixelImageViewport), new FrameworkPropertyMetadata(
                (Brush)null, FrameworkPropertyMetadataOptions.AffectsRender | FrameworkPropertyMetadataOptions.SubPropertiesDoNotAffectRender));

        public static readonly DependencyProperty ForegroundProperty = DependencyProperty.Register(
            "Foreground", typeof(Brush), typeof(PixelImageViewport), new FrameworkPropertyMetadata(
                Brushes.DodgerBlue, FrameworkPropertyMetadataOptions.AffectsRender));

        private WriteableBitmap background = new WriteableBitmap(640, 480, 96.0, 96.0, PixelFormats.Bgra32, null);

        static PixelImageViewport() {

            Style style = CreateDefaultStyles();
            StyleProperty.OverrideMetadata(typeof(PixelImageViewport), new FrameworkPropertyMetadata(style));

            StretchProperty.OverrideMetadata(
                typeof(PixelImageViewport), new FrameworkPropertyMetadata(
                    Stretch.Uniform, FrameworkPropertyMetadataOptions.AffectsMeasure));

            StretchDirectionProperty.OverrideMetadata(
                typeof(PixelImageViewport), new FrameworkPropertyMetadata(
                    StretchDirection.Both, FrameworkPropertyMetadataOptions.AffectsMeasure));
        }

        private static void OnSourceChanged(DependencyObject d, DependencyPropertyChangedEventArgs e) {
            //var image = (PixelImage)d;
            //var oldValue = (ImageSource)e.OldValue;
            //var newValue = (ImageSource)e.NewValue;
        }

        private static Style CreateDefaultStyles() {
            var style = new Style(typeof(PixelImageViewport), null);
            style.Setters.Add(new Setter(FlowDirectionProperty, FlowDirection.LeftToRight));
            style.Seal();
            return style;
        }

        protected override void OnRender(DrawingContext dc) {
            //RenderOptions.SetBitmapScalingMode(this, BitmapScalingMode.NearestNeighbor);
            //dc.DrawImage(background, new Rect(new Point(), RenderSize));

            dc.DrawRectangle(Background, null, new Rect(new Point(), RenderSize));
        }

        protected override Size MeasureOverride(Size constraint) {
            return ComputeSize(constraint);
        }

        protected override Size ArrangeOverride(Size arrangeSize) {
            return ComputeSize(arrangeSize);
        }

        private Size ComputeSize(Size availableSize) {
            var contentSize = new Size(640, 480);
            return MeasureArrangeHelper.ComputeSize(availableSize, contentSize, Stretch, StretchDirection);
        }

    }
}