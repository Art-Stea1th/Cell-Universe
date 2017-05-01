using System;
using System.Security.Cryptography;

namespace ASD.CellUniverse.Infrastructure.DataProviders {

    internal sealed class UniformRandomDataProvider : IDisposable {

        private RNGCryptoServiceProvider crypto;

        public UniformRandomDataProvider() => crypto = new RNGCryptoServiceProvider();

        public byte NextByte() {
            var next = new byte[1];
            crypto.GetBytes(next);
            return next[0];
        }

        public uint NextUint() {
            var next = new byte[4];
            crypto.GetBytes(next);
            return (uint)next[0] << 24 | (uint)next[1] << 16 | (uint)next[2] << 8 | next[3];
        }

        public void Dispose() => crypto?.Dispose();
    }
}