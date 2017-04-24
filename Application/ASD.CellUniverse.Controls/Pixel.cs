using System;
using System.Windows;
using System.Windows.Media;

namespace ASD.CellUniverse.Controls {

    // 8-byte structure (64-bit)
    public struct Pixel {

        // 4-byte
        public short X { get; set; }
        public short Y { get; set; }

        // 4-byte
        public byte A { get; set; }
        public byte R { get; set; }
        public byte G { get; set; }
        public byte B { get; set; }

        public Point Point {
            get => new Point(X, Y);
            set { X = (short)value.X; Y = (short)value.Y; }
        }

        public Color Color {
            get => Color.FromArgb(A, R, G, B);
            set { R = value.R; G = value.G; B = value.B; A = value.A; }
        }

        public byte[] BgraBytes => new byte[4] { B, G, R, A };
        public int BgraInt32 => BitConverter.ToInt32(BgraBytes, 0);

        public static explicit operator Point(Pixel pixel) => pixel.Point;
        public static explicit operator Color(Pixel pixel) => pixel.Color;

        public static explicit operator Pixel(Point point) => From(point);
        public static explicit operator Pixel(Color color) => From(color);

        public static Pixel From(Point point) => new Pixel(point, 255, 0, 0, 0);
        public static Pixel From(Color color) => new Pixel(0, 0, color);

        public Pixel(Point point, Color color)
            : this((short)point.X, (short)point.Y, color.A, color.R, color.G, color.B) { }

        public Pixel(short x, short y, Color color)
            : this(x, y, color.A, color.R, color.G, color.B) { }

        public Pixel(Point point, byte a, byte r, byte g, byte b)
            : this((short)point.X, (short)point.Y, a, r, g, b) { }

        public Pixel(short x, short y, byte a, byte r, byte g, byte b) {
            X = x; Y = y; A = a; R = r; G = g; B = b;
        }
    }
}