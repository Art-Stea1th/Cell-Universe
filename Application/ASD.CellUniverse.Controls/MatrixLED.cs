using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ASD.CellUniverse.Controls {

    using Helpers;
    using Extensions;

    public class MatrixLED : FrameworkElement {

        public byte[,] IntencitySource {
            get => (byte[,])GetValue(IntencitySourceProperty);
            set => SetValue(IntencitySourceProperty, value);
        }

        public Brush Background {
            get => (Brush)GetValue(BackgroundProperty);
            set => SetValue(BackgroundProperty, value);
        }

        public Brush Foreground {
            get => (Brush)GetValue(ForegroundProperty);
            set => SetValue(ForegroundProperty, value);
        }

        public bool ShowGrid {
            get => (bool)GetValue(ShowGridProperty);
            set => SetValue(ShowGridProperty, value);
        }

        public static readonly DependencyProperty IntencitySourceProperty = DependencyProperty.Register(
            "IntencitySource", typeof(byte[,]), typeof(MatrixLED), new FrameworkPropertyMetadata(
                null, FrameworkPropertyMetadataOptions.AffectsMeasure | FrameworkPropertyMetadataOptions.AffectsRender, OnIntencitySourceChanged));

        public static readonly DependencyProperty BackgroundProperty =
            Panel.BackgroundProperty.AddOwner(typeof(MatrixLED), new FrameworkPropertyMetadata(
                Brushes.Transparent, FrameworkPropertyMetadataOptions.AffectsRender | FrameworkPropertyMetadataOptions.SubPropertiesDoNotAffectRender));

        public static readonly DependencyProperty ForegroundProperty =
            TextElement.ForegroundProperty.AddOwner(typeof(MatrixLED), new FrameworkPropertyMetadata(
                SystemColors.ControlTextBrush, FrameworkPropertyMetadataOptions.Inherits));

        public static readonly DependencyProperty ShowGridProperty = DependencyProperty.Register(
            "ShowGrid", typeof(bool), typeof(MatrixLED), new FrameworkPropertyMetadata(
                true, FrameworkPropertyMetadataOptions.AffectsRender));

        static MatrixLED() {
            Style style = CreateDefaultStyles();
            StyleProperty.OverrideMetadata(typeof(MatrixLED), new FrameworkPropertyMetadata(style));
        }

        private static Style CreateDefaultStyles() {
            var style = new Style(typeof(MatrixLED), null);
            style.Setters.Add(new Setter(FlowDirectionProperty, FlowDirection.LeftToRight));
            style.Seal(); return style;
        }

        public MatrixLED() {
            //IntencitySource = buffer = 

            contentSize = CalculateContentSize();
            gridBitmap = BitmapHelper.CreateWriteable(contentSize.X, contentSize.Y);
            gridPixels = GenerateGridMaskPixels(contentSize, cellSize, thickness);

            using (var context = new WriteableContext(gridBitmap)) {
                foreach (var pixel in gridPixels) {
                    context.WriteRect(pixel.x, pixel.y, 1, 1, pixel.color);
                }
            }
        }

        // ---------------


        private Color solidColor = Color.FromArgb(255, 0, 0, 0);
        private Color emptyColor = Color.FromArgb(0, 0, 0, 0);

        private int cellSize = 8;
        private int thickness = 1;

        private ImageBrush GridOpacityMask => new ImageBrush(gridBitmap);
        private WriteableBitmap gridBitmap;
        private IEnumerable<(int x, int y, uint color)> gridPixels;

        private ImageBrush SourceOpacityMask => new ImageBrush(sourceBitmap);
        private WriteableBitmap sourceBitmap;

        private byte[,] buffer = new byte[160, 100];
        //private bool countChanged;

        private SizeInt contentSize;

        private SizeInt CalculateContentSize() {
            var cellsSizeX = buffer.CountX() * cellSize;
            var cellsSizeY = buffer.CountY() * cellSize;
            var thicknessSizeX = buffer.CountX() * thickness + thickness;
            var thisknessSizeY = buffer.CountY() * thickness + thickness;
            return new SizeInt(cellsSizeX + thicknessSizeX, cellsSizeY + thisknessSizeY);
        }

        private IEnumerable<(int x, int y, uint color)> GenerateGridMaskPixels(SizeInt size, int cellSize, int thickness) {
            for (var x = 0; x < size.X; ++x) {
                for (var y = 0; y < size.Y; ++y) {
                    if (x == 0 || y == 0 || x % (cellSize + thickness) == 0 || y % (cellSize + thickness) == 0) {
                        yield return (x: x, y: y, color: solidColor.Bgra32());
                    }
                }
            }
        }


        private static void OnIntencitySourceChanged(DependencyObject d, DependencyPropertyChangedEventArgs e) {

            

            return;

            var prev = e.OldValue as byte[,];
            var next = e.NewValue as byte[,];

            if (next == null || SizeEquals(prev, next)) {
                return;
            }
            //(d as MatrixLED).OnSizeChanged(null);
        }

        private void OnSizeChanged(SizeInt newSize) {

        }

        private static bool SizeEquals<T>(T[,] array1, T[,] array2)
            => array1.GetLength(0) == array2.GetLength(0) && array1.GetLength(1) == array2.GetLength(1);


        // EXPERIMENTS

        protected override void OnRender(DrawingContext dc) {
            RenderOptions.SetBitmapScalingMode(this, BitmapScalingMode.HighQuality);

            //dc.DrawRectangle(Background, null, new Rect(new Point(), RenderSize));

            //dc.PushOpacityMask(SourceOpacityMask);
            //dc.Pop();
            //dc.DrawRectangle(ForegroundOpacityMask, null, new Rect(new Point(), RenderSize));

            //dc.DrawRectangle(Foreground, null, new Rect(new Point(), RenderSize));

            


            dc.DrawImage(gridBitmap, new Rect(new Point(), RenderSize));
        }

        private void DrawClear(DrawingContext dc) {
            dc.DrawRectangle(Background, null, new Rect(new Point(), RenderSize));
        }

        private WriteableBitmap GetSourceBitmap() {

            var random = new Random();

            var colors = new Color[contentSize.X, contentSize.Y];

            for (var y = 0; y < contentSize.Y; y++) {
                for (var x = 0; x < contentSize.X; x++) {
                    colors[x, y] = random.Next() % 2 == 1 ? solidColor : emptyColor;
                    //colors[x, y] = Color.FromArgb(255, 0, 0, 255);
                }
            }

            var bitmap = BitmapHelper.CreateWriteable(contentSize.X, contentSize.Y);
            var buffer = new uint[contentSize.X * contentSize.Y];

            for (var y = 0; y < contentSize.Y; ++y) {
                for (var x = 0; x < contentSize.X; ++x) {
                    var linearIndex = GetLinearIndex(x, y, contentSize.X);
                    buffer[linearIndex] = colors[x, y].Bgra32();
                }
            }

            bitmap.WritePixels(buffer);

            return bitmap;
        }

        private int GetLinearIndex(int x, int y, int width) {
            return width * y + x;
        }

        // END EXPERIMENTS

        protected override Size MeasureOverride(Size constraint) {
            return ComputeSize(constraint);
        }

        protected override Size ArrangeOverride(Size arrangeSize) {
            return ComputeSize(arrangeSize);
        }

        private Size ComputeSize(Size availableSize) {
            return MeasureArrangeHelper.ComputeSize(availableSize, contentSize);
        }

        private struct SizeInt {

            public int X { get; }
            public int Y { get; }

            public SizeInt(int x, int y) { X = x; Y = y; }

            public static explicit operator SizeInt(Size size)
                => new SizeInt((int)size.Width, (int)size.Height);

            public static implicit operator Size(SizeInt size)
                => new Size(size.X, size.Y);
        }
    }
}