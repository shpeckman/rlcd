# src/rlcd/checksum.cr
module RLCD
  class Checksum < IO
    TABLE = StaticArray(UInt32, 256).new do |n|
      crc = n.to_u32
      8.times { crc = crc.bits_set?(1_u32) ? 0xEDB88320_u32 ^ (crc >> 1) : crc >> 1 }
      crc
    end

    @crc : UInt32 = 0xFFFFFFFF_u32

    def initialize(@io : IO)
    end

    def value : UInt32
      ~@crc
    end

    def read(slice : Bytes) : Int32
      count = @io.read(slice)
      update(slice[0, count])
      count
    end

    def write(slice : Bytes) : Nil
      @io.write(slice)
      update(slice)
    end

    private def update(bytes : Bytes) : Nil
      crc = @crc
      bytes.each { |byte| crc = TABLE[(crc ^ byte) & 0xFF_u32] ^ (crc >> 8) }
      @crc = crc
    end
  end
end
